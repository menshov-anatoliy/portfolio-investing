"""Tests for allocation weight functions."""

import numpy as np
import pandas as pd
import pytest

from portfolio_investing.allocation.weights import (
    apply_constraints,
    apply_turnover_cap,
    equal_weight_clusters,
    inverse_vol_clusters,
    risk_parity_clusters,
)


def make_returns(n_assets: int = 6, n_rows: int = 100, seed: int = 0) -> pd.DataFrame:
    """Generate synthetic log returns."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n_assets)]
    return pd.DataFrame(rng.standard_normal((n_rows, n_assets)) * 0.01, columns=tickers)


def make_clusters(n_assets: int = 6) -> dict:
    return {
        1: [f"T{i}" for i in range(0, 2)],
        2: [f"T{i}" for i in range(2, 4)],
        3: [f"T{i}" for i in range(4, n_assets)],
    }


CLUSTERS = make_clusters()
RETURNS = make_returns()


class TestEqualWeightClusters:
    def test_sums_to_one(self):
        w = equal_weight_clusters(CLUSTERS)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)

    def test_non_negative(self):
        w = equal_weight_clusters(CLUSTERS)
        assert all(v >= 0 for v in w.values())

    def test_equal_within_cluster(self):
        w = equal_weight_clusters(CLUSTERS)
        cluster_1_weights = [w["T0"], w["T1"]]
        assert cluster_1_weights[0] == pytest.approx(cluster_1_weights[1], rel=1e-6)

    def test_all_tickers_present(self):
        w = equal_weight_clusters(CLUSTERS)
        assert set(w.keys()) == {f"T{i}" for i in range(6)}


class TestInverseVolClusters:
    def test_sums_to_one(self):
        w = inverse_vol_clusters(CLUSTERS, RETURNS)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)

    def test_non_negative(self):
        w = inverse_vol_clusters(CLUSTERS, RETURNS)
        assert all(v >= 0 for v in w.values())

    def test_all_tickers_present(self):
        w = inverse_vol_clusters(CLUSTERS, RETURNS)
        assert set(w.keys()) == {f"T{i}" for i in range(6)}


class TestRiskParityClusters:
    def test_sums_to_one(self):
        w = risk_parity_clusters(CLUSTERS, RETURNS)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)

    def test_non_negative(self):
        w = risk_parity_clusters(CLUSTERS, RETURNS)
        assert all(v >= 0 for v in w.values())

    def test_all_tickers_present(self):
        w = risk_parity_clusters(CLUSTERS, RETURNS)
        assert set(w.keys()) == {f"T{i}" for i in range(6)}


class TestApplyConstraints:
    def test_max_weight_per_asset(self):
        # One asset has 70% weight; cap each asset at 40% (feasible: 4 × 0.40 ≥ 1.0)
        weights = {"T0": 0.70, "T1": 0.10, "T2": 0.10, "T3": 0.10}
        constrained = apply_constraints(weights, max_weight_per_asset=0.40)
        for v in constrained.values():
            assert v <= 0.40 + 1e-9

    def test_sums_to_one_after_constraint(self):
        weights = {f"T{i}": 1 / 6 for i in range(6)}
        constrained = apply_constraints(weights, max_weight_per_asset=0.10)
        assert sum(constrained.values()) == pytest.approx(1.0, abs=1e-6)

    def test_max_weight_per_cluster(self):
        weights = {"A": 0.5, "B": 0.5}
        clusters = {1: ["A"], 2: ["B"]}
        constrained = apply_constraints(
            weights,
            max_weight_per_asset=1.0,
            max_weight_per_cluster=0.4,
            clusters=clusters,
        )
        # Each cluster capped at 0.4, then renormalized
        assert sum(constrained.values()) == pytest.approx(1.0, abs=1e-6)

    def test_non_negative(self):
        weights = {f"T{i}": 1 / 6 for i in range(6)}
        constrained = apply_constraints(weights)
        assert all(v >= 0 for v in constrained.values())

    def test_cluster_total_respected(self):
        # 3 clusters; one cluster initially holds 60% (2 assets × 30%)
        # Cap cluster at 40% (feasible: 3 clusters × 40% = 120% ≥ 100%)
        weights = {"T0": 0.30, "T1": 0.30, "T2": 0.20, "T3": 0.10, "T4": 0.05, "T5": 0.05}
        clusters = {1: ["T0", "T1"], 2: ["T2", "T3"], 3: ["T4", "T5"]}
        constrained = apply_constraints(
            weights,
            max_weight_per_asset=0.50,
            max_weight_per_cluster=0.40,
            clusters=clusters,
        )
        assert sum(constrained.values()) == pytest.approx(1.0, abs=1e-6)
        cluster1_total = sum(constrained.get(t, 0) for t in clusters[1])
        assert cluster1_total <= 0.40 + 1e-6


class TestApplyTurnoverCap:
    def test_no_cap_when_below_threshold(self):
        curr = {"A": 0.5, "B": 0.5}
        tgt = {"A": 0.52, "B": 0.48}
        result = apply_turnover_cap(curr, tgt, max_turnover=0.30)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)
        # Turnover = 0.02, well below 0.30 -> should move all the way to target
        assert result["A"] == pytest.approx(0.52, abs=1e-6)

    def test_scales_when_above_threshold(self):
        curr = {"A": 0.5, "B": 0.5}
        tgt = {"A": 0.0, "B": 1.0}  # 50% turnover
        result = apply_turnover_cap(curr, tgt, max_turnover=0.20)
        # Should not fully rebalance
        assert result["A"] > 0.0
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)

    def test_sums_to_one(self):
        curr = {"A": 0.3, "B": 0.4, "C": 0.3}
        tgt = {"A": 0.5, "B": 0.1, "C": 0.4}
        result = apply_turnover_cap(curr, tgt, max_turnover=0.15)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)
