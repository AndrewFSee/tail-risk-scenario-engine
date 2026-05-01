"""Risk metrics: VaR, CVaR (Expected Shortfall), drawdown, tail statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TailMetrics:
    var: float            # value-at-risk (loss is positive)
    cvar: float           # expected shortfall
    alpha: float
    mean: float
    std: float
    skew: float
    excess_kurtosis: float


def value_at_risk(pnl: np.ndarray, alpha: float = 0.95) -> float:
    """Historical VaR. ``pnl`` is a return distribution; loss is reported positive."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    q = np.quantile(pnl, 1 - alpha)
    return float(-q)


def conditional_var(pnl: np.ndarray, alpha: float = 0.95) -> float:
    """Conditional VaR / Expected Shortfall at confidence ``alpha``."""
    q = np.quantile(pnl, 1 - alpha)
    tail = pnl[pnl <= q]
    if tail.size == 0:
        return float(-q)
    return float(-tail.mean())


def max_drawdown(cum_returns: np.ndarray) -> float:
    """Max drawdown of a cumulative log-return series. Reported as positive number."""
    eq = np.exp(cum_returns)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(-dd.min())


def tail_metrics(pnl: np.ndarray, alpha: float = 0.95) -> TailMetrics:
    from scipy import stats

    return TailMetrics(
        var=value_at_risk(pnl, alpha),
        cvar=conditional_var(pnl, alpha),
        alpha=alpha,
        mean=float(np.mean(pnl)),
        std=float(np.std(pnl)),
        skew=float(stats.skew(pnl)),
        excess_kurtosis=float(stats.kurtosis(pnl)),
    )
