"""Smoke / integration tests using synthetic data."""

import numpy as np
import pandas as pd
import pytest

from portfolio_investing.backtest.engine import BacktestEngine
from portfolio_investing.reporting.metrics import generate_report, print_report


def make_synthetic_prices(
    n_assets: int = 5,
    n_days: int = 504,  # ~2 years
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic random-walk price data."""
    rng = np.random.default_rng(seed)
    tickers = [f"SYN{i}" for i in range(n_assets)]
    daily_returns = rng.normal(0.0003, 0.01, size=(n_days, n_assets))
    prices = 100.0 * np.exp(np.cumsum(daily_returns, axis=0))
    dates = pd.bdate_range(start="2021-01-04", periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=tickers)


MINIMAL_CONFIG = {
    "correlation": {
        "rolling_window": 63,
        "min_periods": 30,
        "shrinkage": True,
    },
    "clustering": {
        "n_clusters_min": 2,
        "n_clusters_max": 3,
        "linkage": "ward",
    },
    "allocation": {
        "max_weight_per_asset": 0.50,
        "max_weight_per_cluster": 0.80,
    },
    "rebalance": {
        "threshold_pct": 0.05,
    },
    "costs": {
        "commission_pct": 0.001,
        "slippage_pct": 0.001,
    },
    "constraints": {
        "max_turnover_per_period": 0.30,
    },
    "reporting": {
        "risk_free_rate": 0.05,
    },
}

PRICES = make_synthetic_prices()


class TestSmokeBacktest:
    def test_portfolio_starts_at_one(self):
        engine = BacktestEngine(MINIMAL_CONFIG)
        result = engine.run(PRICES, "equal_weight", "monthly")
        pv = result["portfolio_values"]
        assert pv.iloc[0] == pytest.approx(1.0, abs=1e-10)

    def test_portfolio_length(self):
        engine = BacktestEngine(MINIMAL_CONFIG)
        result = engine.run(PRICES, "equal_weight", "monthly")
        pv = result["portfolio_values"]
        # prices -> returns loses 1 row; engine keeps (returns_len - min_periods + 1) rows
        min_periods = MINIMAL_CONFIG["correlation"]["min_periods"]
        # prices -> returns loses 1 row (via dropna); engine emits (returns_len - min_periods + 1) values
        expected_len = len(PRICES) - min_periods  # equivalent to len(returns) - min_periods + 1
        assert len(pv) == expected_len

    def test_no_nan_in_portfolio(self):
        engine = BacktestEngine(MINIMAL_CONFIG)
        result = engine.run(PRICES, "equal_weight", "monthly")
        pv = result["portfolio_values"]
        assert not pv.isna().any()

    def test_positive_portfolio_values(self):
        engine = BacktestEngine(MINIMAL_CONFIG)
        result = engine.run(PRICES, "equal_weight", "monthly")
        pv = result["portfolio_values"]
        assert (pv > 0).all()

    def test_rebalance_dates_not_empty(self):
        engine = BacktestEngine(MINIMAL_CONFIG)
        result = engine.run(PRICES, "equal_weight", "monthly")
        assert len(result["rebalance_dates"]) > 0

    def test_weights_history_not_empty(self):
        engine = BacktestEngine(MINIMAL_CONFIG)
        result = engine.run(PRICES, "equal_weight", "monthly")
        assert len(result["weights_history"]) > 0

    def test_cluster_history_not_empty(self):
        engine = BacktestEngine(MINIMAL_CONFIG)
        result = engine.run(PRICES, "equal_weight", "monthly")
        assert len(result["cluster_history"]) > 0


class TestSmokeReport:
    def test_report_generates_without_error(self):
        engine = BacktestEngine(MINIMAL_CONFIG)
        results = {}
        for alloc in ("equal_weight", "inverse_vol"):
            for reb in ("monthly", "quarterly"):
                result = engine.run(PRICES, alloc, reb)
                results[(alloc, reb)] = result
        report = generate_report(results, MINIMAL_CONFIG)
        assert report is not None
        assert len(report) == 4

    def test_report_has_expected_columns(self):
        engine = BacktestEngine(MINIMAL_CONFIG)
        result = engine.run(PRICES, "equal_weight", "monthly")
        report = generate_report({("equal_weight", "monthly"): result}, MINIMAL_CONFIG)
        for col in ("cagr", "sharpe", "max_drawdown"):
            assert col in report.columns

    def test_print_report_runs(self, capsys):
        engine = BacktestEngine(MINIMAL_CONFIG)
        result = engine.run(PRICES, "equal_weight", "monthly")
        report = generate_report({("equal_weight", "monthly"): result}, MINIMAL_CONFIG)
        print_report(report)
        captured = capsys.readouterr()
        assert "Portfolio Strategy Comparison" in captured.out


class TestSmokeAllocationMethods:
    @pytest.mark.parametrize("alloc", ["equal_weight", "inverse_vol", "risk_parity"])
    def test_allocation_method_runs(self, alloc):
        engine = BacktestEngine(MINIMAL_CONFIG)
        result = engine.run(PRICES, alloc, "monthly")
        pv = result["portfolio_values"]
        assert not pv.isna().any()
        assert pv.iloc[0] == pytest.approx(1.0, abs=1e-10)
