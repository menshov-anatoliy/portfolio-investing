from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as exc:  # pragma: no cover
    raise ImportError("matplotlib and seaborn are required. Install with: pip install matplotlib seaborn") from exc
try:
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
except ImportError as exc:  # pragma: no cover
    raise ImportError("scipy is required. Install with: pip install scipy") from exc

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError("yfinance is required. Install with: pip install yfinance") from exc


MOEX_BASE = "https://iss.moex.com/iss"
HISTORY_PAGE_SIZE = 100
MAX_CAP_ITERATIONS = 20


@dataclass(frozen=True)
class MoexInstrument:
    secid: str
    board: str
    market: str
    alias: str | None = None

    @property
    def name(self) -> str:
        return self.alias or self.secid


MOEX_SHARES: List[MoexInstrument] = [
    MoexInstrument("SBERP", "TQBR", "shares"),
    MoexInstrument("ZAYM", "TQBR", "shares"),
    MoexInstrument("SIBN", "TQBR", "shares"),
    MoexInstrument("BELU", "TQBR", "shares"),
    MoexInstrument("TATNP", "TQBR", "shares"),
    MoexInstrument("PLZL", "TQBR", "shares"),
    MoexInstrument("PHOR", "TQBR", "shares"),
    MoexInstrument("MOEX", "TQBR", "shares"),
    MoexInstrument("MDMG", "TQBR", "shares"),
    MoexInstrument("LKOH", "TQBR", "shares"),
]

MOEX_FUNDS_BONDS: List[MoexInstrument] = [
    MoexInstrument("RU000A104KU3", "TQCB", "bonds", alias="PARUS-LOG"),
    MoexInstrument("RENT", "TQIF", "shares", alias="RENT"),
]

CRYPTO_MARKET_TICKERS = ["BTC-USD", "TAO-USD", "ETH-USD", "USDT-USD"]
SEARCH_QUERIES = ["ПАРУС", "Рентал ПРО", "ДОМ.РФ"]


def search_moex_securities(query: str, limit: int = 10) -> pd.DataFrame:
    url = f"{MOEX_BASE}/securities.json"
    response = requests.get(url, params={"q": query}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    sec = payload.get("securities", {})
    columns = sec.get("columns", [])
    data = sec.get("data", [])[:limit]
    return pd.DataFrame(data, columns=columns)


def fetch_moex_history(instr: MoexInstrument, start_date: str, end_date: str) -> pd.Series:
    all_rows: List[List[object]] = []
    columns: List[str] = []
    start = 0

    while True:
        url = (
            f"{MOEX_BASE}/history/engines/stock/markets/{instr.market}"
            f"/boards/{instr.board}/securities/{instr.secid}.json"
        )
        try:
            response = requests.get(
                url,
                params={"from": start_date, "till": end_date, "start": start},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            warnings.warn(f"MOEX request failed for {instr.name}: {exc}", stacklevel=2)
            return pd.Series(dtype=float, name=instr.name)
        payload = response.json()
        block = payload.get("history", {})
        if not columns:
            columns = block.get("columns", [])
        page_data = block.get("data", [])
        if not page_data:
            break
        all_rows.extend(page_data)
        if len(page_data) < HISTORY_PAGE_SIZE:
            break
        start += HISTORY_PAGE_SIZE

    if not all_rows:
        return pd.Series(dtype=float, name=instr.name)

    df = pd.DataFrame(all_rows, columns=columns)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    close_col = "CLOSE" if "CLOSE" in df.columns else "LEGALCLOSEPRICE"
    price = pd.to_numeric(df[close_col], errors="coerce")
    series = pd.Series(price.values, index=df["TRADEDATE"], name=instr.name).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return series


def fetch_crypto_history(ticker: str, start_date: str, end_date: str) -> pd.Series:
    try:
        hist = yf.download(
            ticker,
            start=start_date,
            end=(pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            interval="1d",
        )
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"yfinance request failed for {ticker}: {exc}", stacklevel=2)
        return pd.Series(dtype=float, name=ticker)
    if hist.empty:
        return pd.Series(dtype=float, name=ticker)
    if isinstance(hist.columns, pd.MultiIndex):
        close = hist["Close"][ticker]
    else:
        close = hist["Close"]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.rename(ticker).sort_index()


def cap_weights(weights: pd.Series, max_weight: float) -> pd.Series:
    w = weights.copy().astype(float)
    for _ in range(MAX_CAP_ITERATIONS):
        above = w[w > max_weight]
        if above.empty:
            break
        overflow = float((above - max_weight).sum())
        w.loc[above.index] = max_weight
        below = w[w < max_weight]
        if below.empty:
            break
        redistribute = below / below.sum()
        w.loc[below.index] += overflow * redistribute
    return w / w.sum()


def correl_dist(corr: pd.DataFrame) -> pd.DataFrame:
    # HRP angular distance transform: d(i,j) = sqrt((1-rho(i,j))/2).
    corr = corr.clip(-1, 1)
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist.values, 0.0)
    return dist


def get_quasi_diag(link: np.ndarray) -> List[int]:
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix.loc[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def get_cluster_var(cov: pd.DataFrame, items: List[str]) -> float:
    cov_ = cov.loc[items, items]
    ivp = 1.0 / np.diag(cov_.values)
    ivp /= ivp.sum()
    w = ivp.reshape(-1, 1)
    return float(w.T @ cov_.values @ w)


def split_cluster_once(cluster: List[str]) -> List[List[str]]:
    if len(cluster) <= 1:
        return []
    mid = len(cluster) // 2
    return [cluster[:mid], cluster[mid:]]


def hrp_allocation(cov: pd.DataFrame, corr: pd.DataFrame, method: str = "ward") -> Tuple[pd.Series, np.ndarray, List[str]]:
    dist = correl_dist(corr)
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method=method)
    sorted_idx = get_quasi_diag(link)
    ordered_assets = corr.index[sorted_idx].tolist()
    weights = pd.Series(1.0, index=ordered_assets)
    clusters = [ordered_assets]
    while clusters:
        next_clusters: List[List[str]] = []
        for cluster in clusters:
            next_clusters.extend(split_cluster_once(cluster))
        clusters = next_clusters
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            var0 = get_cluster_var(cov, c0)
            var1 = get_cluster_var(cov, c1)
            alpha = 1 - var0 / (var0 + var1)
            weights[c0] *= alpha
            weights[c1] *= 1 - alpha
    return weights / weights.sum(), link, ordered_assets


def build_price_matrix(start_date: str, end_date: str) -> pd.DataFrame:
    moex_series: Dict[str, pd.Series] = {}
    for instr in MOEX_SHARES + MOEX_FUNDS_BONDS:
        s = fetch_moex_history(instr, start_date, end_date)
        if s.empty:
            warnings.warn(f"MOEX instrument has no data: {instr.name} ({instr.secid})", stacklevel=2)
            continue
        moex_series[instr.name] = s

    if not moex_series:
        raise RuntimeError(
            "No MOEX data loaded. Check network access to iss.moex.com and verify tickers/boards/market mapping."
        )

    moex_df = pd.concat(moex_series.values(), axis=1).sort_index()
    moex_df.columns = list(moex_series.keys())
    moex_calendar = moex_df.index.unique().sort_values()

    crypto_series: Dict[str, pd.Series] = {}
    for ticker in CRYPTO_MARKET_TICKERS:
        s = fetch_crypto_history(ticker, start_date, end_date)
        if s.empty:
            warnings.warn(f"Crypto ticker has no data: {ticker}", stacklevel=2)
            continue
        crypto_series[ticker] = s

    if not crypto_series:
        raise RuntimeError("No crypto data loaded. Check yfinance availability/tickers.")

    crypto_df = pd.concat(crypto_series.values(), axis=1).sort_index()
    crypto_df.columns = list(crypto_series.keys())
    crypto_on_moex = crypto_df.reindex(moex_calendar).ffill()

    full_df = pd.concat([moex_df, crypto_on_moex], axis=1).sort_index().ffill()
    return full_df


def determine_common_window(prices: pd.DataFrame) -> pd.DataFrame:
    first_valid = prices.apply(pd.Series.first_valid_index)
    if first_valid.isna().any():
        missing = first_valid[first_valid.isna()].index.tolist()
        raise RuntimeError(f"No valid observations for assets: {missing}")

    common_start = max(first_valid)
    trimmed = prices.loc[common_start:].dropna(how="any")
    if trimmed.empty:
        raise RuntimeError("No overlapping date window after synchronization.")
    return trimmed


def warn_short_history(prices: pd.DataFrame, min_days: int = 126) -> List[str]:
    short_assets: List[str] = []
    for col in prices.columns:
        obs = int(prices[col].dropna().shape[0])
        if obs < min_days:
            short_assets.append(col)
            warnings.warn(
                f"Asset {col} has only {obs} observations (<{min_days}). "
                f"Consider excluding it or trimming all assets to its start date.",
                stacklevel=2,
            )
    return short_assets


def run_backtest(prices: pd.DataFrame, target_weights: pd.Series, threshold: float = 0.10) -> pd.DataFrame:
    returns = prices.pct_change().fillna(0.0)
    assets = target_weights.index.tolist()
    portfolio_value = 1.0
    holdings = target_weights * portfolio_value
    rows: List[Dict[str, float]] = []
    rebalances = 0

    for dt, ret in returns[assets].iterrows():
        holdings *= (1.0 + ret)
        portfolio_value = float(holdings.sum())
        current_weights = holdings / portfolio_value
        drift = (current_weights - target_weights).abs()
        do_rebalance = bool((drift > threshold).any())
        if do_rebalance:
            holdings = target_weights * portfolio_value
            current_weights = target_weights.copy()
            rebalances += 1
        row = {"date": dt, "portfolio_value": portfolio_value, "rebalanced": float(do_rebalance)}
        row.update({f"w_{a}": float(current_weights[a]) for a in assets})
        rows.append(row)

    out = pd.DataFrame(rows).set_index("date")
    out.attrs["rebalances"] = rebalances
    return out


def save_plots(
    corr: pd.DataFrame,
    link: np.ndarray,
    labels: List[str],
    ordered_assets: List[str],
    weights: pd.Series,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 9))
    sns.heatmap(corr.loc[ordered_assets, ordered_assets], annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    dendrogram(link, labels=labels, leaf_rotation=90)
    plt.title("HRP Dendrogram")
    plt.tight_layout()
    plt.savefig(out_dir / "dendrogram.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 9))
    plt.pie(weights.values, labels=weights.index, autopct="%1.1f%%", startangle=90)
    plt.title("HRP Weights")
    plt.tight_layout()
    plt.savefig(out_dir / "hrp_weights_pie.png", dpi=150)
    plt.close()


def run(
    start_date: str,
    end_date: str,
    out_dir: Path,
    max_weight: float = 0.15,
    rebalance_threshold: float = 0.10,
) -> None:
    print("=== MOEX ticker discovery ===")
    for query in SEARCH_QUERIES:
        try:
            df = search_moex_securities(query)
            print(f"\nQuery: {query}")
            if df.empty:
                print("  No matches")
            else:
                cols = [c for c in ["SECID", "SHORTNAME", "BOARDID", "PRIMARY_BOARDID"] if c in df.columns]
                print(df[cols].head(10).to_string(index=False))
        except Exception as exc:  # pragma: no cover
            print(f"  Search failed for '{query}': {exc}")

    print("\n=== Data loading and synchronization ===")
    prices = build_price_matrix(start_date, end_date)
    short_check_assets = [a for a in ["ZAYM", "RENT"] if a in prices.columns]
    if short_check_assets:
        warn_short_history(prices[short_check_assets].dropna(how="all"), min_days=126)
    prices = determine_common_window(prices)
    print(f"Common synchronized window: {prices.index.min().date()} .. {prices.index.max().date()}")
    print(f"Assets used ({len(prices.columns)}): {', '.join(prices.columns)}")

    log_returns = np.log(prices / prices.shift(1)).dropna(how="any")
    corr = log_returns.corr()
    cov = log_returns.cov()

    weights, link, ordered_assets = hrp_allocation(cov, corr, method="ward")
    if max_weight > 0:
        weights = cap_weights(weights, max_weight=max_weight)
    weights = weights.sort_values(ascending=False)

    backtest = run_backtest(prices, weights, threshold=rebalance_threshold)
    total_return = backtest["portfolio_value"].iloc[-1] - 1.0
    rebalances = int(backtest.attrs.get("rebalances", 0))

    save_plots(corr, link, list(corr.index), ordered_assets, weights, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_table = pd.DataFrame({"asset": weights.index, "weight": weights.values, "weight_pct": weights.values * 100})
    weights_table.to_csv(out_dir / "hrp_weights.csv", index=False)
    backtest.to_csv(out_dir / "backtest_timeseries.csv")

    print("\n=== HRP Weights (%) ===")
    print(weights_table[["asset", "weight_pct"]].to_string(index=False, formatters={"weight_pct": "{:.2f}".format}))
    print(f"\nBacktest total return: {total_return:.2%}")
    print(f"Rebalances triggered: {rebalances}")
    print(f"Output directory: {out_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HRP portfolio for MOEX + crypto with calendar synchronization.")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=date.today().isoformat(), help="End date YYYY-MM-DD")
    parser.add_argument("--out-dir", default="output", help="Directory for charts and tables")
    parser.add_argument("--max-weight", type=float, default=0.15, help="Maximum weight per asset after HRP")
    parser.add_argument(
        "--rebalance-threshold",
        type=float,
        default=0.10,
        help="Absolute drift threshold for rebalancing trigger",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        start_date=args.start,
        end_date=args.end,
        out_dir=Path(args.out_dir),
        max_weight=args.max_weight,
        rebalance_threshold=args.rebalance_threshold,
    )
