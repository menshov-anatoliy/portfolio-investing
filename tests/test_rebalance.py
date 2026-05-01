"""Tests for rebalance trigger logic."""

import pandas as pd
import pytest

from portfolio_investing.rebalance.rebalancer import (
    needs_rebalance,
    needs_rebalance_calendar,
    needs_rebalance_threshold,
)


class TestCalendarRebalance:
    def test_monthly_same_month_false(self):
        curr = pd.Timestamp("2023-01-15")
        last = pd.Timestamp("2023-01-01")
        assert needs_rebalance_calendar(curr, last, "monthly") is False

    def test_monthly_different_month_true(self):
        curr = pd.Timestamp("2023-02-01")
        last = pd.Timestamp("2023-01-31")
        assert needs_rebalance_calendar(curr, last, "monthly") is True

    def test_monthly_different_year_true(self):
        curr = pd.Timestamp("2024-01-01")
        last = pd.Timestamp("2023-12-31")
        assert needs_rebalance_calendar(curr, last, "monthly") is True

    def test_quarterly_same_quarter_false(self):
        curr = pd.Timestamp("2023-03-15")
        last = pd.Timestamp("2023-01-01")
        assert needs_rebalance_calendar(curr, last, "quarterly") is False

    def test_quarterly_different_quarter_true(self):
        curr = pd.Timestamp("2023-04-01")
        last = pd.Timestamp("2023-03-31")
        assert needs_rebalance_calendar(curr, last, "quarterly") is True

    def test_quarterly_same_quarter_end(self):
        curr = pd.Timestamp("2023-06-30")
        last = pd.Timestamp("2023-04-01")
        assert needs_rebalance_calendar(curr, last, "quarterly") is False

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            needs_rebalance_calendar(
                pd.Timestamp("2023-01-01"), pd.Timestamp("2022-12-01"), "daily"
            )


class TestThresholdRebalance:
    def test_small_deviation_false(self):
        curr = {"A": 0.50, "B": 0.50}
        tgt = {"A": 0.52, "B": 0.48}
        assert needs_rebalance_threshold(curr, tgt, threshold_pct=0.05) is False

    def test_large_deviation_true(self):
        curr = {"A": 0.60, "B": 0.40}
        tgt = {"A": 0.50, "B": 0.50}
        assert needs_rebalance_threshold(curr, tgt, threshold_pct=0.05) is True

    def test_exact_threshold_not_triggered(self):
        # deviation < threshold -> not triggered
        curr = {"A": 0.54, "B": 0.46}
        tgt = {"A": 0.50, "B": 0.50}
        assert needs_rebalance_threshold(curr, tgt, threshold_pct=0.05) is False

    def test_new_asset_in_target(self):
        curr = {"A": 1.0}
        tgt = {"A": 0.50, "B": 0.50}
        assert needs_rebalance_threshold(curr, tgt, threshold_pct=0.05) is True


class TestNeedsRebalance:
    def test_monthly_mode(self):
        assert needs_rebalance(
            pd.Timestamp("2023-02-01"),
            pd.Timestamp("2023-01-31"),
            {"A": 0.5},
            {"A": 0.5},
            mode="monthly",
        ) is True

    def test_quarterly_mode(self):
        assert needs_rebalance(
            pd.Timestamp("2023-04-01"),
            pd.Timestamp("2023-03-31"),
            {"A": 0.5},
            {"A": 0.5},
            mode="quarterly",
        ) is True

    def test_threshold_mode(self):
        assert needs_rebalance(
            pd.Timestamp("2023-01-15"),
            pd.Timestamp("2023-01-01"),
            {"A": 0.60},
            {"A": 0.50},
            mode="threshold",
            threshold_pct=0.05,
        ) is True

    def test_hybrid_monthly_both_conditions(self):
        # Calendar: different month; Threshold: deviation > 5%
        result = needs_rebalance(
            pd.Timestamp("2023-02-01"),
            pd.Timestamp("2023-01-31"),
            {"A": 0.60},
            {"A": 0.50},
            mode="hybrid_monthly",
            threshold_pct=0.05,
        )
        assert result is True

    def test_hybrid_monthly_only_calendar(self):
        # Calendar: different month; Threshold: deviation <= 5% -> no rebalance
        result = needs_rebalance(
            pd.Timestamp("2023-02-01"),
            pd.Timestamp("2023-01-31"),
            {"A": 0.52},
            {"A": 0.50},
            mode="hybrid_monthly",
            threshold_pct=0.05,
        )
        assert result is False

    def test_hybrid_quarterly_both_conditions(self):
        result = needs_rebalance(
            pd.Timestamp("2023-04-01"),
            pd.Timestamp("2023-03-31"),
            {"A": 0.60},
            {"A": 0.50},
            mode="hybrid_quarterly",
            threshold_pct=0.05,
        )
        assert result is True

    def test_hybrid_quarterly_only_threshold(self):
        # Same quarter, large deviation -> no rebalance (need both)
        result = needs_rebalance(
            pd.Timestamp("2023-02-15"),
            pd.Timestamp("2023-01-01"),
            {"A": 0.70},
            {"A": 0.50},
            mode="hybrid_quarterly",
            threshold_pct=0.05,
        )
        assert result is False

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            needs_rebalance(
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2022-12-01"),
                {},
                {},
                mode="bogus",
            )
