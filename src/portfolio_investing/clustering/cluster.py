"""Hierarchical clustering of assets based on correlation distances."""

import logging

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


def cluster_assets(
    distance_matrix: pd.DataFrame,
    n_clusters: int,
    linkage_method: str = "ward",
) -> dict:
    """
    Cluster assets using hierarchical clustering.

    Parameters
    ----------
    distance_matrix : pd.DataFrame
        Symmetric distance matrix (tickers x tickers).
    n_clusters : int
        Number of clusters to form.
    linkage_method : str
        Linkage criterion for scipy (e.g. "ward", "average").

    Returns
    -------
    dict
        Mapping {cluster_id (int): [list of ticker strings]}.
    """
    tickers = list(distance_matrix.columns)
    n = len(tickers)

    if n_clusters >= n:
        n_clusters = n
        logger.warning("n_clusters clamped to n_assets=%d.", n)

    dist_arr = distance_matrix.values.copy()
    np.fill_diagonal(dist_arr, 0.0)
    condensed = squareform(dist_arr, checks=False)

    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    clusters: dict = {}
    for ticker, cluster_id in zip(tickers, labels):
        clusters.setdefault(int(cluster_id), []).append(ticker)

    logger.debug("Formed %d clusters from %d assets.", len(clusters), n)
    return clusters


def select_n_clusters(
    distance_matrix: pd.DataFrame,
    min_k: int = 3,
    max_k: int = 5,
    linkage_method: str = "ward",
) -> int:
    """
    Select optimal number of clusters using silhouette score.

    Parameters
    ----------
    distance_matrix : pd.DataFrame
        Symmetric distance matrix.
    min_k : int
        Minimum number of clusters to try.
    max_k : int
        Maximum number of clusters to try.
    linkage_method : str
        Linkage criterion.

    Returns
    -------
    int
        Optimal number of clusters in [min_k, max_k].
    """
    n = len(distance_matrix)
    max_k = min(max_k, n - 1)
    min_k = min(min_k, max_k)

    tickers = list(distance_matrix.columns)
    dist_arr = distance_matrix.values.copy()
    np.fill_diagonal(dist_arr, 0.0)
    condensed = squareform(dist_arr, checks=False)
    Z = linkage(condensed, method=linkage_method)

    best_k = min_k
    best_score = -np.inf

    for k in range(min_k, max_k + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(dist_arr, labels, metric="precomputed")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Silhouette failed for k=%d: %s", k, exc)
            continue
        logger.debug("k=%d silhouette=%.4f", k, score)
        if score > best_score:
            best_score = score
            best_k = k

    logger.info("Selected n_clusters=%d (silhouette=%.4f).", best_k, best_score)
    return best_k


def compute_intercluster_correlation(
    corr_matrix: pd.DataFrame,
    clusters: dict,
) -> pd.DataFrame:
    """
    Compute average pairwise correlation between clusters.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Full correlation matrix.
    clusters : dict
        Mapping {cluster_id: [tickers]}.

    Returns
    -------
    pd.DataFrame
        Cluster-level average correlation matrix.
    """
    cluster_ids = sorted(clusters.keys())
    n = len(cluster_ids)
    inter = np.zeros((n, n))

    for i, ci in enumerate(cluster_ids):
        for j, cj in enumerate(cluster_ids):
            assets_i = clusters[ci]
            assets_j = clusters[cj]
            sub = corr_matrix.loc[assets_i, assets_j]
            inter[i, j] = float(sub.values.mean())

    return pd.DataFrame(
        inter,
        index=cluster_ids,
        columns=cluster_ids,
    )


def check_cluster_stability(cluster_history: list) -> dict:
    """
    Measure stability of cluster assignments over time.

    Parameters
    ----------
    cluster_history : list
        List of cluster dicts {cluster_id: [tickers]}, one per period.

    Returns
    -------
    dict
        Keys: 'mean_overlap', 'min_overlap', 'summary'.
    """
    if len(cluster_history) < 2:
        return {"mean_overlap": 1.0, "min_overlap": 1.0, "summary": "Only one period."}

    def assignment_map(clusters: dict) -> dict:
        return {ticker: cid for cid, tickers in clusters.items() for ticker in tickers}

    overlaps = []
    for prev, curr in zip(cluster_history[:-1], cluster_history[1:]):
        prev_map = assignment_map(prev)
        curr_map = assignment_map(curr)
        common = set(prev_map) & set(curr_map)
        if not common:
            overlaps.append(0.0)
            continue
        same = sum(1 for t in common if prev_map[t] == curr_map[t])
        overlaps.append(same / len(common))

    mean_overlap = float(np.mean(overlaps))
    min_overlap = float(np.min(overlaps))
    summary = (
        f"Cluster stability over {len(cluster_history)} periods: "
        f"mean_overlap={mean_overlap:.3f}, min_overlap={min_overlap:.3f}"
    )
    logger.info(summary)
    return {"mean_overlap": mean_overlap, "min_overlap": min_overlap, "summary": summary}
