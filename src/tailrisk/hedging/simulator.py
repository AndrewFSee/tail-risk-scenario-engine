"""Compare hedged vs. unhedged portfolio outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..risk.metrics import conditional_var, value_at_risk
from .instruments import InverseETF, PutOption


@dataclass
class HedgeComparison:
    name: str
    expected_return: float
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float


def evaluate_hedges(
    asset_step_returns: np.ndarray,
    weights: np.ndarray,
    hedges: list[PutOption | InverseETF],
) -> list[HedgeComparison]:
    """Return per-hedge metrics including the unhedged baseline."""
    asset_terminal = asset_step_returns.sum(axis=1)            # log -> additive
    base_pnl = asset_terminal @ weights

    results = [_summarize("unhedged", base_pnl)]
    for h in hedges:
        if isinstance(h, PutOption):
            extra = h.payoff(asset_terminal)
            label = f"put_{h.moneyness:.2f}m"
        elif isinstance(h, InverseETF):
            extra = h.payoff_from_steps(asset_step_returns)
            label = f"inverse_etf_{h.allocation:.2%}"
        else:  # pragma: no cover
            raise TypeError(f"Unknown hedge type: {type(h)}")
        results.append(_summarize(label, base_pnl + extra))
    return results


def _summarize(name: str, pnl: np.ndarray) -> HedgeComparison:
    return HedgeComparison(
        name=name,
        expected_return=float(pnl.mean()),
        var_95=value_at_risk(pnl, 0.95),
        cvar_95=conditional_var(pnl, 0.95),
        var_99=value_at_risk(pnl, 0.99),
        cvar_99=conditional_var(pnl, 0.99),
    )
