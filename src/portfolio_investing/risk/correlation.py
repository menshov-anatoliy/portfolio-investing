"""Correlation and distance matrix computations."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices.

    Returns
    -------
    pd.DataFrame
        Log return DataFrame.
    """
    return np.log(prices / prices.shift(1)).dropna()


def _apply_shrinkage(returns: pd.DataFrame) -> pd.DataFrame:
    """Apply Ledoit-Wolf shrinkage to returns and return a correlation matrix."""
    lw = LedoitWolf()
    lw.fit(returns)
    cov = lw.covariance_
    std = np.sqrt(np.diag(cov))
    # Avoid division by zero for near-constant series
    std = np.where(std == 0, 1e-10, std)
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    return pd.DataFrame(corr, index=returns.columns, columns=returns.columns)


def compute_correlation(
    returns: pd.DataFrame,
    window: Optional[int] = None,
    shrinkage: bool = True,
    min_periods: int = 63,
) -> pd.DataFrame:
    """
    Compute correlation matrix from returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Log returns.
    window : int, optional
        If provided, use only the last ``window`` rows.
    shrinkage : bool
        Apply Ledoit-Wolf shrinkage when True.
    min_periods : int
        Minimum number of valid observations required.

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    if window is not None:
        data = returns.iloc[-window:]
    else:
        data = returns

    if len(data) < min_periods:
        logger.warning(
            "Only %d periods available (min_periods=%d); using available data.",
            len(data),
            min_periods,
        )

    if shrinkage and len(data) >= 2:
        try:
            return _apply_shrinkage(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ledoit-Wolf shrinkage failed (%s); falling back to sample corr.", exc)

    corr = data.corr()
    return corr


def compute_rolling_correlations(
    returns: pd.DataFrame,
    window: int,
    min_periods: int = 63,
) -> dict:
    """
    Compute rolling correlation matrices for each valid date.

    Parameters
    ----------
    returns : pd.DataFrame
        Log returns.
    window : int
        Rolling window in trading days.
    min_periods : int
        Minimum valid observations in window.

    Returns
    -------
    dict
        Mapping of date -> correlation matrix DataFrame.
    """
    result = {}
    dates = returns.index
    for i, date in enumerate(dates):
        if i < min_periods - 1:
            continue
        start = max(0, i - window + 1)
        chunk = returns.iloc[start : i + 1]
        if len(chunk) < min_periods:
            continue
        try:
            corr = compute_correlation(chunk, shrinkage=True, min_periods=min_periods)
            result[date] = corr
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping date %s: %s", date, exc)
    logger.debug("Computed rolling correlations for %d dates.", len(result))
    return result


def correlation_distance(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distance matrix from a correlation matrix.

    Uses the formula: d_ij = sqrt(0.5 * (1 - rho_ij)).

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix.

    Returns
    -------
    pd.DataFrame
        Distance matrix with same index/columns.
    """
    dist = np.sqrt(0.5 * (1.0 - corr_matrix.values))
    dist = np.clip(dist, 0.0, None)
    return pd.DataFrame(dist, index=corr_matrix.index, columns=corr_matrix.columns)
