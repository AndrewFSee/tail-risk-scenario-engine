"""CVaR portfolio optimization (Rockafellar–Uryasev linear program).

Given a scenario matrix of asset returns ``R`` of shape (S, N) and confidence
level alpha, choose weights w to minimize portfolio CVaR_alpha:

    minimize    eta + (1 / ((1 - alpha) * S)) * sum_s u_s
    subject to  u_s >= -R_s @ w - eta,    u_s >= 0
                w in W   (e.g. simplex)

The objective is convex in (w, eta, u) and solved via cvxpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CVaROptResult:
    weights: np.ndarray
    cvar: float
    var: float
    expected_return: float


def optimize_cvar(
    scenario_returns: np.ndarray,
    alpha: float = 0.95,
    long_only: bool = True,
    max_weight: float = 1.0,
    target_return: float | None = None,
) -> CVaROptResult:
    """Solve the Rockafellar–Uryasev CVaR LP and return optimal weights.

    Parameters
    ----------
    scenario_returns:
        Shape (S, N) — terminal scenario returns per asset.
    alpha:
        Confidence level, e.g. 0.95.
    long_only:
        If True, weights >= 0.
    max_weight:
        Per-asset cap.
    target_return:
        Optional minimum expected portfolio return constraint.
    """
    import cvxpy as cp

    R = np.asarray(scenario_returns)
    S, N = R.shape

    w = cp.Variable(N)
    eta = cp.Variable()
    u = cp.Variable(S, nonneg=True)

    portfolio_loss = -R @ w  # losses are positive
    constraints = [
        u >= portfolio_loss - eta,
        cp.sum(w) == 1,
        w <= max_weight,
    ]
    if long_only:
        constraints.append(w >= 0)
    if target_return is not None:
        mu = R.mean(axis=0)
        constraints.append(mu @ w >= target_return)

    cvar_expr = eta + (1.0 / ((1.0 - alpha) * S)) * cp.sum(u)
    prob = cp.Problem(cp.Minimize(cvar_expr), constraints)
    prob.solve()

    if w.value is None:
        raise RuntimeError(f"CVaR optimization failed: status={prob.status}")

    w_opt = np.asarray(w.value).flatten()
    pnl = R @ w_opt
    var_alpha = float(-np.quantile(pnl, 1 - alpha))
    tail = pnl[pnl <= -var_alpha]
    cvar = float(-tail.mean()) if tail.size else var_alpha
    return CVaROptResult(
        weights=w_opt,
        cvar=cvar,
        var=var_alpha,
        expected_return=float(pnl.mean()),
    )
