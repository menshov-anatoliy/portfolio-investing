"""Performance metrics and reporting utilities."""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


def compute_cagr(portfolio_values: pd.Series) -> float:
    """
    Compute compound annual growth rate.

    Parameters
    ----------
    portfolio_values : pd.Series
        Daily portfolio values starting at 1.0.

    Returns
    -------
    float
        CAGR as a decimal (e.g. 0.12 for 12%).
    """
    if len(portfolio_values) < 2:
        return 0.0
    years = len(portfolio_values) / _TRADING_DAYS
    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0]
    return float(total_return ** (1.0 / years) - 1.0)


def compute_sharpe(
    portfolio_values: pd.Series,
    risk_free_rate: float = 0.10,
) -> float:
    """
    Compute annualized Sharpe ratio.

    Parameters
    ----------
    portfolio_values : pd.Series
        Daily portfolio values.
    risk_free_rate : float
        Annual risk-free rate (decimal).

    Returns
    -------
    float
        Sharpe ratio.
    """
    daily_returns = portfolio_values.pct_change().dropna()
    if len(daily_returns) < 2:
        return 0.0
    rf_daily = (1 + risk_free_rate) ** (1 / _TRADING_DAYS) - 1
    excess = daily_returns - rf_daily
    std = excess.std()
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(_TRADING_DAYS))


def compute_sortino(
    portfolio_values: pd.Series,
    risk_free_rate: float = 0.10,
) -> float:
    """
    Compute annualized Sortino ratio.

    Parameters
    ----------
    portfolio_values : pd.Series
        Daily portfolio values.
    risk_free_rate : float
        Annual risk-free rate (decimal).

    Returns
    -------
    float
        Sortino ratio.
    """
    daily_returns = portfolio_values.pct_change().dropna()
    if len(daily_returns) < 2:
        return 0.0
    rf_daily = (1 + risk_free_rate) ** (1 / _TRADING_DAYS) - 1
    excess = daily_returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = downside.std()
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(_TRADING_DAYS))


def compute_max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Compute maximum drawdown.

    Parameters
    ----------
    portfolio_values : pd.Series
        Daily portfolio values.

    Returns
    -------
    float
        Maximum drawdown as a positive decimal (e.g. 0.25 for 25%).
    """
    rolling_max = portfolio_values.cummax()
    drawdowns = (portfolio_values - rolling_max) / rolling_max
    return float(drawdowns.min())


def compute_turnover(turnover_history: list) -> float:
    """
    Compute average turnover per rebalance period.

    Parameters
    ----------
    turnover_history : list
        List of (date, turnover_pct) tuples.

    Returns
    -------
    float
        Mean turnover.
    """
    if not turnover_history:
        return 0.0
    return float(np.mean([t for _, t in turnover_history]))


def generate_report(results: dict, config: dict) -> pd.DataFrame:
    """
    Generate comparison report across strategy combinations.

    Parameters
    ----------
    results : dict
        Mapping of {(allocation_method, rebalance_mode): backtest_result}.
    config : dict
        Configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Report with strategy combinations as rows and metrics as columns.
    """
    rf = config.get("reporting", {}).get("risk_free_rate", 0.10)
    rows = []
    for (alloc, reb), result in results.items():
        pv = result["portfolio_values"]
        row = {
            "allocation": alloc,
            "rebalance": reb,
            "cagr": compute_cagr(pv),
            "sharpe": compute_sharpe(pv, rf),
            "sortino": compute_sortino(pv, rf),
            "max_drawdown": compute_max_drawdown(pv),
            "avg_turnover": compute_turnover(result.get("turnover_history", [])),
            "n_rebalances": len(result.get("rebalance_dates", [])),
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index(["allocation", "rebalance"])
    return df


def print_report(report_df: pd.DataFrame) -> None:
    """
    Pretty-print the report DataFrame to console.

    Parameters
    ----------
    report_df : pd.DataFrame
        Report from ``generate_report``.
    """
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print("\n=== Portfolio Strategy Comparison ===")
    print(report_df.to_string())
    print()


def save_report(report_df: pd.DataFrame, output_dir: str) -> None:
    """
    Save report as CSV to output directory.

    Parameters
    ----------
    report_df : pd.DataFrame
        Report from ``generate_report``.
    output_dir : str
        Directory to save the CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "strategy_report.csv")
    report_df.to_csv(path)
    logger.info("Report saved to %s", path)
