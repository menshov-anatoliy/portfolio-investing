"""Tests for asset clustering."""

import numpy as np
import pandas as pd
import pytest

from portfolio_investing.clustering.cluster import (
    check_cluster_stability,
    cluster_assets,
    compute_intercluster_correlation,
    select_n_clusters,
)
from portfolio_investing.risk.correlation import correlation_distance


def make_distance_matrix(n: int = 8, seed: int = 42) -> pd.DataFrame:
    """Generate a valid symmetric distance matrix."""
    rng = np.random.default_rng(seed)
    # Build synthetic returns and compute real distances
    tickers = [f"T{i}" for i in range(n)]
    returns = pd.DataFrame(rng.standard_normal((200, n)), columns=tickers)
    corr = returns.corr()
    return correlation_distance(corr)


class TestClusterAssets:
    def test_exact_n_clusters(self):
        dist = make_distance_matrix(8)
        clusters = cluster_assets(dist, n_clusters=3)
        assert len(clusters) == 3

    def test_all_assets_appear_once(self):
        dist = make_distance_matrix(8)
        clusters = cluster_assets(dist, n_clusters=4)
        all_tickers = []
        for tickers in clusters.values():
            all_tickers.extend(tickers)
        assert sorted(all_tickers) == sorted(dist.columns.tolist())
        # No duplicates
        assert len(all_tickers) == len(set(all_tickers))

    def test_deterministic(self):
        dist = make_distance_matrix(10)
        c1 = cluster_assets(dist, n_clusters=3)
        c2 = cluster_assets(dist, n_clusters=3)
        # Same cluster membership (may have different cluster IDs)
        members1 = sorted([sorted(v) for v in c1.values()])
        members2 = sorted([sorted(v) for v in c2.values()])
        assert members1 == members2

    def test_single_cluster(self):
        dist = make_distance_matrix(5)
        clusters = cluster_assets(dist, n_clusters=1)
        all_tickers = [t for tickers in clusters.values() for t in tickers]
        assert sorted(all_tickers) == sorted(dist.columns.tolist())

    def test_n_clusters_equals_n_assets(self):
        dist = make_distance_matrix(4)
        clusters = cluster_assets(dist, n_clusters=4)
        assert sum(len(v) for v in clusters.values()) == 4


class TestSelectNClusters:
    def test_returns_in_range(self):
        dist = make_distance_matrix(10)
        k = select_n_clusters(dist, min_k=2, max_k=5)
        assert 2 <= k <= 5

    def test_min_k_equals_max_k(self):
        dist = make_distance_matrix(10)
        k = select_n_clusters(dist, min_k=3, max_k=3)
        assert k == 3

    def test_small_dataset(self):
        dist = make_distance_matrix(4)
        k = select_n_clusters(dist, min_k=2, max_k=3)
        assert 2 <= k <= 3


class TestInterclusterCorrelation:
    def test_shape(self):
        dist = make_distance_matrix(6)
        clusters = cluster_assets(dist, n_clusters=3)
        # Build fake corr matrix matching dist columns
        tickers = list(dist.columns)
        corr = pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
        inter = compute_intercluster_correlation(corr, clusters)
        assert inter.shape == (3, 3)

    def test_diagonal_is_one_for_single_asset_clusters(self):
        # With single-asset clusters, intra-cluster = corr[i,i] = 1.0
        n = 3
        tickers = [f"T{i}" for i in range(n)]
        corr = pd.DataFrame(np.eye(n), index=tickers, columns=tickers)
        single_clusters = {i + 1: [t] for i, t in enumerate(tickers)}
        inter = compute_intercluster_correlation(corr, single_clusters)
        for i in range(n):
            assert inter.iloc[i, i] == pytest.approx(1.0, abs=1e-10)


class TestClusterStability:
    def test_single_period(self):
        result = check_cluster_stability([{1: ["A", "B"], 2: ["C"]}])
        assert result["mean_overlap"] == 1.0

    def test_identical_periods(self):
        clusters = {1: ["A", "B"], 2: ["C", "D"]}
        result = check_cluster_stability([clusters, clusters, clusters])
        assert result["mean_overlap"] == pytest.approx(1.0)

    def test_completely_different(self):
        c1 = {1: ["A", "B"], 2: ["C", "D"]}
        c2 = {1: ["C", "D"], 2: ["A", "B"]}
        result = check_cluster_stability([c1, c2])
        # All assets switched clusters
        assert result["mean_overlap"] == pytest.approx(0.0)

    def test_returns_summary_string(self):
        result = check_cluster_stability([{1: ["A"]}, {1: ["A"]}])
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0
