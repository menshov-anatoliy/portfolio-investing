"""Tests for correlation distance computation."""

import math

import numpy as np
import pandas as pd
import pytest

from portfolio_investing.risk.correlation import (
    compute_correlation,
    compute_returns,
    correlation_distance,
)


def make_corr(rho_val: float, n: int = 3) -> pd.DataFrame:
    """Make a constant-correlation matrix."""
    mat = np.full((n, n), rho_val)
    np.fill_diagonal(mat, 1.0)
    tickers = [f"A{i}" for i in range(n)]
    return pd.DataFrame(mat, index=tickers, columns=tickers)


class TestCorrelationDistance:
    def test_symmetric(self):
        corr = make_corr(0.5)
        dist = correlation_distance(corr)
        pd.testing.assert_frame_equal(dist, dist.T)

    def test_zero_diagonal(self):
        corr = make_corr(0.3)
        dist = correlation_distance(corr)
        for i in range(len(dist)):
            assert dist.iloc[i, i] == pytest.approx(0.0, abs=1e-10)

    def test_range_non_negative(self):
        corr = make_corr(-0.8, n=4)
        dist = correlation_distance(corr)
        assert (dist.values >= 0).all()

    def test_perfect_correlation_zero_distance(self):
        corr = make_corr(1.0)
        dist = correlation_distance(corr)
        # Off-diagonal should be 0 (rho=1 -> d=0)
        assert dist.iloc[0, 1] == pytest.approx(0.0, abs=1e-10)

    def test_perfect_negative_correlation(self):
        corr = make_corr(-1.0)
        dist = correlation_distance(corr)
        # rho=-1 -> d = sqrt(0.5*(1-(-1))) = sqrt(1) = 1
        assert dist.iloc[0, 1] == pytest.approx(1.0, abs=1e-10)

    def test_zero_correlation(self):
        corr = make_corr(0.0)
        dist = correlation_distance(corr)
        # rho=0 -> d = sqrt(0.5) ≈ 0.7071
        expected = math.sqrt(0.5)
        assert dist.iloc[0, 1] == pytest.approx(expected, abs=1e-6)

    def test_within_sqrt2_range(self):
        corr = make_corr(-0.5, n=5)
        dist = correlation_distance(corr)
        assert (dist.values <= math.sqrt(2) + 1e-10).all()


class TestComputeReturns:
    def test_log_returns_shape(self):
        prices = pd.DataFrame(
            {"A": [100, 101, 99], "B": [50, 52, 51]},
            index=pd.date_range("2021-01-01", periods=3),
        )
        returns = compute_returns(prices)
        assert returns.shape == (2, 2)

    def test_log_returns_values(self):
        prices = pd.DataFrame(
            {"A": [100.0, 110.0]},
            index=pd.date_range("2021-01-01", periods=2),
        )
        returns = compute_returns(prices)
        expected = math.log(110.0 / 100.0)
        assert returns["A"].iloc[0] == pytest.approx(expected, rel=1e-6)
