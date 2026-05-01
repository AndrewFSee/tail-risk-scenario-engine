"""Hedging instrument payoff models.

Simplified, *engine-internal* models — not exchange-grade pricing. Intended
for *relative* comparison between hedge variants on simulated paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PutOption:
    """Long European put. Payoff at expiry vs. underlying terminal return.

    The underlying is identified by index ``asset_index`` in the simulated
    asset-return tensor. Strike is expressed as moneyness (K = moneyness * S0).
    """

    asset_index: int
    moneyness: float = 0.95
    premium_pct: float = 0.012        # premium paid up front, as fraction of notional
    notional_fraction: float = 1.0    # notional as fraction of portfolio value

    def payoff(self, asset_terminal_returns: np.ndarray) -> np.ndarray:
        """Returns net payoff per simulated path as a fraction of portfolio value."""
        s_t_over_s0 = np.exp(asset_terminal_returns[:, self.asset_index])
        payoff = np.maximum(self.moneyness - s_t_over_s0, 0.0)
        return self.notional_fraction * (payoff - self.premium_pct)


@dataclass
class InverseETF:
    """Static short-equity sleeve approximated as -1x the underlying return.

    A real -1x ETF has daily-rebalancing decay; we model that approximately
    by accumulating step-by-step rather than using terminal return.
    """

    asset_index: int
    allocation: float = 0.05

    def payoff_from_steps(self, asset_step_returns: np.ndarray) -> np.ndarray:
        """asset_step_returns: (n_paths, horizon, n_assets) log-returns."""
        r = asset_step_returns[:, :, self.asset_index]
        # daily rebalanced -1x: product of (1 - r_t) - 1
        compounded = np.prod(1.0 - r, axis=1) - 1.0
        return self.allocation * compounded
