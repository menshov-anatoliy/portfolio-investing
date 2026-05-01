"""Portfolio weight allocation methods."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-10


def _normalize(weights: dict) -> dict:
    """Normalize weight dict so values sum to 1.0."""
    total = sum(weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def equal_weight_clusters(
    clusters: dict,
    prices: pd.DataFrame = None,
) -> dict:
    """
    Assign equal weight to each cluster; equal weight within each cluster.

    Parameters
    ----------
    clusters : dict
        Mapping {cluster_id: [tickers]}.
    prices : pd.DataFrame, optional
        Not used; included for API consistency.

    Returns
    -------
    dict
        {ticker: weight} summing to 1.0.
    """
    n_clusters = len(clusters)
    if n_clusters == 0:
        return {}

    cluster_weight = 1.0 / n_clusters
    weights: dict = {}
    for tickers in clusters.values():
        if not tickers:
            continue
        per_asset = cluster_weight / len(tickers)
        for ticker in tickers:
            weights[ticker] = per_asset

    return _normalize(weights)


def inverse_vol_clusters(
    clusters: dict,
    returns: pd.DataFrame,
    window: int = 63,
) -> dict:
    """
    Weight clusters by inverse of cluster-average return volatility.
    Within each cluster, equal weight.

    Parameters
    ----------
    clusters : dict
        Mapping {cluster_id: [tickers]}.
    returns : pd.DataFrame
        Log returns.
    window : int
        Rolling window for volatility estimation.

    Returns
    -------
    dict
        {ticker: weight} summing to 1.0.
    """
    recent = returns.iloc[-window:] if len(returns) > window else returns
    cluster_vols: dict = {}
    for cid, tickers in clusters.items():
        valid = [t for t in tickers if t in recent.columns]
        if not valid:
            cluster_vols[cid] = _EPS
            continue
        avg_ret = recent[valid].mean(axis=1)
        vol = avg_ret.std()
        cluster_vols[cid] = max(vol, _EPS)

    inv_vol = {cid: 1.0 / v for cid, v in cluster_vols.items()}
    total_inv = sum(inv_vol.values())
    cluster_weights = {cid: v / total_inv for cid, v in inv_vol.items()}

    weights: dict = {}
    for cid, tickers in clusters.items():
        valid = [t for t in tickers if t in recent.columns]
        if not valid:
            continue
        per_asset = cluster_weights.get(cid, 0.0) / len(valid)
        for ticker in valid:
            weights[ticker] = per_asset

    return _normalize(weights)


def risk_parity_clusters(
    clusters: dict,
    returns: pd.DataFrame,
    window: int = 63,
    max_iter: int = 100,
) -> dict:
    """
    Equal risk contribution (ERC) across clusters; equal weight within cluster.

    Parameters
    ----------
    clusters : dict
        Mapping {cluster_id: [tickers]}.
    returns : pd.DataFrame
        Log returns.
    window : int
        Rolling window for covariance estimation.
    max_iter : int
        Iterations for ERC solver.

    Returns
    -------
    dict
        {ticker: weight} summing to 1.0.
    """
    recent = returns.iloc[-window:] if len(returns) > window else returns
    cluster_ids = sorted(clusters.keys())
    n = len(cluster_ids)

    # Build cluster-level return series
    cluster_returns = pd.DataFrame()
    for cid in cluster_ids:
        tickers = [t for t in clusters[cid] if t in recent.columns]
        if tickers:
            cluster_returns[cid] = recent[tickers].mean(axis=1)
        else:
            cluster_returns[cid] = 0.0

    cov = cluster_returns.cov().values.copy()
    cov += np.eye(n) * _EPS  # regularize

    # ERC via iterative algo
    w = np.ones(n) / n
    for _ in range(max_iter):
        sigma_p = np.sqrt(w @ cov @ w)
        mrc = cov @ w / (sigma_p + _EPS)
        rc = w * mrc
        w_new = w * (1.0 / (n * rc + _EPS))
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < 1e-8:
            break
        w = w_new

    w = np.abs(w_new)  # clip any floating-point negatives to zero
    w = np.maximum(w, 0.0)
    w /= w.sum() if w.sum() > 0 else 1.0

    cluster_weights = {cid: float(w[i]) for i, cid in enumerate(cluster_ids)}

    weights: dict = {}
    for cid in cluster_ids:
        tickers = [t for t in clusters[cid] if t in recent.columns]
        if not tickers:
            continue
        per_asset = cluster_weights[cid] / len(tickers)
        for ticker in tickers:
            weights[ticker] = per_asset

    return _normalize(weights)


def apply_constraints(
    weights: dict,
    max_weight_per_asset: float = 0.20,
    max_weight_per_cluster: float = 0.50,
    clusters: dict = None,
) -> dict:
    """
    Apply weight constraints and renormalize.

    Parameters
    ----------
    weights : dict
        {ticker: weight} from an allocation function.
    max_weight_per_asset : float
        Maximum weight allowed for a single asset.
    max_weight_per_cluster : float
        Maximum total weight allowed for a single cluster.
    clusters : dict, optional
        Required for cluster-level constraint.

    Returns
    -------
    dict
        Constrained and renormalized weights summing to 1.0.
    """
    # Clip individual asset weights iteratively until convergence
    w = {k: max(v, 0.0) for k, v in weights.items()}
    for _ in range(100):
        w = {k: min(v, max_weight_per_asset) for k, v in w.items()}
        total = sum(w.values())
        if total <= 0:
            n_w = len(w)
            w = {k: 1.0 / n_w for k in w}
            break
        w = {k: v / total for k, v in w.items()}
        if all(v <= max_weight_per_asset + 1e-10 for v in w.values()):
            break

    # Clip cluster-level weights iteratively
    if clusters is not None:
        for _ in range(100):
            changed = False
            for _cid, tickers in clusters.items():
                cluster_total = sum(w.get(t, 0.0) for t in tickers)
                if cluster_total > max_weight_per_cluster + 1e-10:
                    scale = max_weight_per_cluster / cluster_total
                    for t in tickers:
                        if t in w:
                            w[t] *= scale
                    changed = True
            total = sum(w.values())
            if total > 0:
                w = {k: v / total for k, v in w.items()}
            if not changed:
                break

    return w


def apply_turnover_cap(
    current_weights: dict,
    target_weights: dict,
    max_turnover: float = 0.30,
) -> dict:
    """
    Scale trades so total turnover does not exceed max_turnover.

    Parameters
    ----------
    current_weights : dict
        Current portfolio weights {ticker: weight}.
    target_weights : dict
        Target weights after rebalance {ticker: weight}.
    max_turnover : float
        Maximum allowed turnover (sum of absolute trades).

    Returns
    -------
    dict
        Adjusted new weights {ticker: weight}, summing to 1.0.
    """
    all_tickers = set(current_weights) | set(target_weights)
    curr = {t: current_weights.get(t, 0.0) for t in all_tickers}
    tgt = {t: target_weights.get(t, 0.0) for t in all_tickers}

    total_turnover = sum(abs(tgt[t] - curr[t]) for t in all_tickers) / 2.0
    if total_turnover <= max_turnover or total_turnover < _EPS:
        return _normalize(tgt)

    scale = max_turnover / total_turnover
    new_weights = {t: curr[t] + scale * (tgt[t] - curr[t]) for t in all_tickers}
    new_weights = {k: max(v, 0.0) for k, v in new_weights.items()}
    return _normalize(new_weights)
