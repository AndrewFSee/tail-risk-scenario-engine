"""Per-asset attribution of tail loss.

Computes how much each asset contributes to the portfolio's CVaR by averaging
asset-level returns *conditional on* the portfolio being in its left tail.
"""

from __future__ import annotations

import numpy as np


def cvar_contribution(
    asset_returns_terminal: np.ndarray,  # (n_paths, n_assets) terminal return per asset
    weights: np.ndarray,
    alpha: float = 0.95,
) -> np.ndarray:
    """Component CVaR contributions, summing to portfolio CVaR.

    For each asset j: contrib_j = -w_j * E[R_j | R_p <= VaR_alpha].
    """
    portfolio = asset_returns_terminal @ weights
    threshold = np.quantile(portfolio, 1 - alpha)
    mask = portfolio <= threshold
    if not mask.any():
        return np.zeros_like(weights)
    cond_means = asset_returns_terminal[mask].mean(axis=0)
    return -weights * cond_means
