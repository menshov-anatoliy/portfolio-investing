"""Rebalance trigger logic."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def needs_rebalance_calendar(
    current_date: pd.Timestamp,
    last_rebalance_date: pd.Timestamp,
    mode: str,
) -> bool:
    """
    Determine if a calendar-based rebalance is due.

    Parameters
    ----------
    current_date : pd.Timestamp
        Today's date.
    last_rebalance_date : pd.Timestamp
        Date of the last rebalance.
    mode : str
        "monthly" or "quarterly".

    Returns
    -------
    bool
        True if rebalance should occur.
    """
    if mode == "monthly":
        return (
            current_date.year != last_rebalance_date.year
            or current_date.month != last_rebalance_date.month
        )
    elif mode == "quarterly":
        return (
            current_date.year != last_rebalance_date.year
            or current_date.quarter != last_rebalance_date.quarter
        )
    else:
        raise ValueError(f"Unknown calendar mode: {mode!r}")


def needs_rebalance_threshold(
    current_weights: dict,
    target_weights: dict,
    threshold_pct: float = 0.05,
) -> bool:
    """
    Check if any asset has drifted beyond threshold from target.

    Parameters
    ----------
    current_weights : dict
        Current portfolio weights {ticker: weight}.
    target_weights : dict
        Target weights {ticker: weight}.
    threshold_pct : float
        Maximum allowed absolute deviation before rebalance.

    Returns
    -------
    bool
        True if rebalance threshold is breached.
    """
    all_tickers = set(current_weights) | set(target_weights)
    for ticker in all_tickers:
        deviation = abs(
            current_weights.get(ticker, 0.0) - target_weights.get(ticker, 0.0)
        )
        if deviation >= threshold_pct:
            return True
    return False


def needs_rebalance(
    current_date,
    last_rebalance_date,
    current_weights: dict,
    target_weights: dict,
    mode: str,
    threshold_pct: float = 0.05,
) -> bool:
    """
    Unified rebalance decision for all modes.

    Parameters
    ----------
    current_date : pd.Timestamp
        Today's date.
    last_rebalance_date : pd.Timestamp
        Date of the last rebalance.
    current_weights : dict
        Current drifted weights.
    target_weights : dict
        Target weights.
    mode : str
        One of: "monthly", "quarterly", "threshold",
        "hybrid_monthly", "hybrid_quarterly".
    threshold_pct : float
        Deviation threshold for threshold/hybrid modes.

    Returns
    -------
    bool
        True if portfolio should be rebalanced.
    """
    current_date = pd.Timestamp(current_date)
    last_rebalance_date = pd.Timestamp(last_rebalance_date)

    if mode in ("monthly", "quarterly"):
        return needs_rebalance_calendar(current_date, last_rebalance_date, mode)

    if mode == "threshold":
        return needs_rebalance_threshold(current_weights, target_weights, threshold_pct)

    if mode == "hybrid_monthly":
        return needs_rebalance_calendar(
            current_date, last_rebalance_date, "monthly"
        ) and needs_rebalance_threshold(current_weights, target_weights, threshold_pct)

    if mode == "hybrid_quarterly":
        return needs_rebalance_calendar(
            current_date, last_rebalance_date, "quarterly"
        ) and needs_rebalance_threshold(current_weights, target_weights, threshold_pct)

    raise ValueError(f"Unknown rebalance mode: {mode!r}")
