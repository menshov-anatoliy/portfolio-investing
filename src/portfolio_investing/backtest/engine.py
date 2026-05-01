"""Walk-forward backtesting engine."""

import logging
from typing import Any

import numpy as np
import pandas as pd

from portfolio_investing.allocation.weights import (
    apply_constraints,
    apply_turnover_cap,
    equal_weight_clusters,
    inverse_vol_clusters,
    risk_parity_clusters,
)
from portfolio_investing.clustering.cluster import cluster_assets, select_n_clusters
from portfolio_investing.rebalance.rebalancer import needs_rebalance
from portfolio_investing.risk.correlation import (
    compute_correlation,
    compute_returns,
    correlation_distance,
)

logger = logging.getLogger(__name__)

_ALLOCATION_METHODS = {
    "equal_weight": equal_weight_clusters,
    "inverse_vol": inverse_vol_clusters,
    "risk_parity": risk_parity_clusters,
}


class BacktestEngine:
    """Walk-forward portfolio backtesting engine."""

    def __init__(self, config: dict):
        """
        Initialize engine with configuration.

        Parameters
        ----------
        config : dict
            Full config dictionary.
        """
        self.config = config
        self.corr_cfg = config.get("correlation", {})
        self.cluster_cfg = config.get("clustering", {})
        self.alloc_cfg = config.get("allocation", {})
        self.reb_cfg = config.get("rebalance", {})
        self.cost_cfg = config.get("costs", {})
        self.constraint_cfg = config.get("constraints", {})

    def _compute_weights(
        self,
        returns: pd.DataFrame,
        allocation_method: str,
    ) -> tuple[dict, dict]:
        """Compute cluster assignments and portfolio weights from returns."""
        window = self.corr_cfg.get("rolling_window", 126)
        shrinkage = self.corr_cfg.get("shrinkage", True)
        min_periods = self.corr_cfg.get("min_periods", 63)
        linkage = self.cluster_cfg.get("linkage", "ward")
        min_k = self.cluster_cfg.get("n_clusters_min", 3)
        max_k = self.cluster_cfg.get("n_clusters_max", 5)

        corr = compute_correlation(
            returns,
            window=window,
            shrinkage=shrinkage,
            min_periods=min_periods,
        )
        dist = correlation_distance(corr)

        n_clusters = select_n_clusters(dist, min_k=min_k, max_k=max_k, linkage_method=linkage)
        clusters = cluster_assets(dist, n_clusters=n_clusters, linkage_method=linkage)

        alloc_fn = _ALLOCATION_METHODS.get(allocation_method, equal_weight_clusters)
        if allocation_method in ("inverse_vol", "risk_parity"):
            raw_weights = alloc_fn(clusters, returns)
        else:
            raw_weights = alloc_fn(clusters)

        max_asset = self.alloc_cfg.get("max_weight_per_asset", 0.20)
        max_cluster = self.alloc_cfg.get("max_weight_per_cluster", 0.50)
        weights = apply_constraints(
            raw_weights,
            max_weight_per_asset=max_asset,
            max_weight_per_cluster=max_cluster,
            clusters=clusters,
        )
        return clusters, weights

    def run(
        self,
        prices: pd.DataFrame,
        allocation_method: str,
        rebalance_mode: str,
    ) -> dict:
        """
        Run walk-forward backtest.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices indexed by date.
        allocation_method : str
            One of "equal_weight", "inverse_vol", "risk_parity".
        rebalance_mode : str
            One of "monthly", "quarterly", "threshold",
            "hybrid_monthly", "hybrid_quarterly".

        Returns
        -------
        dict
            'portfolio_values', 'weights_history', 'rebalance_dates',
            'turnover_history', 'cluster_history'.
        """
        returns = compute_returns(prices)
        min_periods = self.corr_cfg.get("min_periods", 63)
        max_turnover = self.constraint_cfg.get("max_turnover_per_period", 0.30)
        threshold_pct = self.reb_cfg.get("threshold_pct", 0.05)
        commission = self.cost_cfg.get("commission_pct", 0.001)
        slippage = self.cost_cfg.get("slippage_pct", 0.001)
        total_cost_pct = commission + slippage

        dates = returns.index
        n = len(dates)

        if n < min_periods:
            raise ValueError(
                f"Insufficient data: {n} return rows, need {min_periods}."
            )

        # Initial cluster + weights using first min_periods of data
        init_returns = returns.iloc[:min_periods]
        clusters, target_weights = self._compute_weights(init_returns, allocation_method)
        current_weights = dict(target_weights)

        portfolio_value = 1.0
        portfolio_values: list[float] = []
        weights_history: list[tuple] = [(dates[min_periods - 1], dict(current_weights))]
        rebalance_dates: list = [dates[min_periods - 1]]
        turnover_history: list[tuple] = []
        cluster_history: list[tuple] = [(dates[min_periods - 1], clusters)]
        last_rebalance_date = dates[min_periods - 1]

        for i in range(min_periods, n):
            date = dates[i]
            prev_date = dates[i - 1]

            # Drift current weights by today's returns
            daily_ret = returns.loc[date]
            asset_returns = {t: daily_ret.get(t, 0.0) for t in current_weights}
            new_values = {t: current_weights[t] * (1 + np.expm1(asset_returns[t])) for t in current_weights}
            total = sum(new_values.values())
            if total > 0:
                current_weights = {t: v / total for t, v in new_values.items()}
                portfolio_value *= total
            else:
                portfolio_value *= 1.0

            portfolio_values.append(portfolio_value)

            # Check rebalance trigger
            if needs_rebalance(
                current_date=date,
                last_rebalance_date=last_rebalance_date,
                current_weights=current_weights,
                target_weights=target_weights,
                mode=rebalance_mode,
                threshold_pct=threshold_pct,
            ):
                # Recompute cluster + target using recent history
                hist_returns = returns.loc[:date]
                try:
                    clusters, new_target = self._compute_weights(hist_returns, allocation_method)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Weight computation failed at %s: %s – reusing old.", date, exc)
                    new_target = target_weights

                # Align weights to available tickers
                available = set(prices.columns)
                current_aligned = {t: current_weights.get(t, 0.0) for t in available}
                target_aligned = {t: new_target.get(t, 0.0) for t in available}

                # Apply turnover cap
                adjusted = apply_turnover_cap(current_aligned, target_aligned, max_turnover)

                # Compute realized turnover
                turnover = sum(
                    abs(adjusted.get(t, 0.0) - current_aligned.get(t, 0.0))
                    for t in available
                ) / 2.0

                # Apply costs
                cost = turnover * total_cost_pct
                portfolio_value *= 1.0 - cost

                current_weights = adjusted
                target_weights = new_target
                last_rebalance_date = date

                rebalance_dates.append(date)
                turnover_history.append((date, turnover))
                weights_history.append((date, dict(current_weights)))
                cluster_history.append((date, clusters))

        port_series = pd.Series(
            [1.0] + portfolio_values,
            index=pd.Index([dates[min_periods - 1]] + list(dates[min_periods:])),
        )

        logger.info(
            "Backtest done: %d days, %d rebalances, final value=%.4f",
            len(port_series),
            len(rebalance_dates),
            port_series.iloc[-1],
        )

        return {
            "portfolio_values": port_series,
            "weights_history": weights_history,
            "rebalance_dates": rebalance_dates,
            "turnover_history": turnover_history,
            "cluster_history": cluster_history,
        }
