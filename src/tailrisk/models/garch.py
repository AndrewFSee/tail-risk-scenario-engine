"""GARCH(1,1) volatility model wrapper around the ``arch`` package.

We keep the wrapper minimal: fit on a return series, expose conditional volatility,
and provide a forecast generator suitable for path simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GarchFit:
    """A fitted GARCH(1,1) with Student-t innovations."""

    omega: float
    alpha: float
    beta: float
    nu: float                 # innovation degrees of freedom
    last_variance: float      # h_T
    last_resid: float         # epsilon_T (standardized * sqrt(h_T))
    mean: float               # constant mean

    def simulate_variance_path(
        self,
        horizon: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate ``horizon`` steps of (returns, variances) forward from the fit.

        Returns are mean + sqrt(h_t) * z_t with z_t ~ standardized Student-t.
        """
        from scipy import stats

        h = np.empty(horizon)
        r = np.empty(horizon)
        h_prev = self.last_variance
        eps_prev = self.last_resid
        # standardized t with unit variance
        t_scale = np.sqrt((self.nu - 2.0) / self.nu) if self.nu > 2 else 1.0
        for t in range(horizon):
            h_t = self.omega + self.alpha * eps_prev**2 + self.beta * h_prev
            z = stats.t.rvs(self.nu, size=1, random_state=rng)[0] * t_scale
            eps_t = np.sqrt(h_t) * z
            r[t] = self.mean + eps_t
            h[t] = h_t
            h_prev = h_t
            eps_prev = eps_t
        return r, h


def fit_garch(returns: pd.Series | np.ndarray) -> GarchFit:
    """Fit GARCH(1,1) with Student-t innovations using the ``arch`` package."""
    from arch import arch_model

    r = pd.Series(np.asarray(returns)).dropna()
    # arch expects returns in percent for numerical stability.
    am = arch_model(r * 100.0, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(disp="off")
    p = res.params
    # Convert percent-scale back to decimal-scale variance.
    omega = float(p["omega"]) / 1e4
    alpha = float(p["alpha[1]"])
    beta = float(p["beta[1]"])
    nu = float(p["nu"])
    mean = float(p["mu"]) / 100.0
    last_variance = float(res.conditional_volatility.iloc[-1] ** 2) / 1e4
    last_resid = float(res.resid.iloc[-1]) / 100.0
    return GarchFit(
        omega=omega,
        alpha=alpha,
        beta=beta,
        nu=nu,
        last_variance=last_variance,
        last_resid=last_resid,
        mean=mean,
    )
