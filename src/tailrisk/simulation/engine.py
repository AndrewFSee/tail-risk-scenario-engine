"""Monte Carlo simulation engine.

Composes marginal, volatility, regime, and dependence models into per-asset
return paths, then aggregates to portfolio P&L.

The default pipeline is:

    1. Fit per-asset GARCH(1,1)-t   -> conditional volatility paths
    2. Fit cross-asset Student-t copula on standardized residuals
       -> dependence structure with tail co-movement
    3. (Optional) Fit 2-state Markov regime model on the portfolio series
       -> in crisis days, scale variance and floor correlation
    4. Draw uniforms from the copula, push through marginal quantile,
       multiply by simulated conditional vol, add mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..config import EngineConfig
from ..models import (
    GarchFit,
    GaussianCopula,
    RegimeFit,
    StudentTCopula,
    StudentTFit,
    fit_garch,
    fit_markov_2state,
    fit_student_t,
)


@dataclass
class FittedModels:
    tickers: list[str]
    marginals: list[StudentTFit]
    garch: list[GarchFit] | None
    copula: GaussianCopula | StudentTCopula
    regime: RegimeFit | None
    # Optional regime-conditional copulas (one per regime). When set, the
    # engine picks copula by simulated regime state at each timestep, modeling
    # correlation breakdowns during stress.
    copula_by_regime: list[GaussianCopula | StudentTCopula] | None = None
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class SimulationResult:
    asset_paths: np.ndarray   # (n_paths, horizon, n_assets) cumulative returns
    asset_returns: np.ndarray  # (n_paths, horizon, n_assets) per-step returns
    portfolio_pnl: np.ndarray  # (n_paths,) terminal portfolio return
    weights: np.ndarray
    tickers: list[str]


def fit_models(returns: pd.DataFrame, cfg: EngineConfig) -> FittedModels:
    """Fit all configured models on a returns panel."""
    tickers = list(returns.columns)
    marginals = [fit_student_t(returns[c].values) for c in tickers]
    garch = (
        [fit_garch(returns[c]) for c in tickers] if cfg.simulation.volatility == "garch" else None
    )
    if cfg.simulation.dependence == "student_t_copula":
        cop: GaussianCopula | StudentTCopula = StudentTCopula.fit(returns.values)
    else:
        cop = GaussianCopula.fit(returns.values)
    regime = (
        fit_markov_2state(returns.mean(axis=1))
        if cfg.simulation.regime == "markov_2state"
        else None
    )

    # Regime-conditional copulas: classify each historical day by its smoothed
    # regime probability and fit a separate copula on each subset. Only do this
    # when both classes have enough observations for a stable correlation
    # estimate.
    copula_by_regime: list[GaussianCopula | StudentTCopula] | None = None
    if regime is not None:
        labels = regime.smoothed_probs.argmax(axis=1)
        # Align labels (which were fit on the portfolio-mean series, same length
        # as `returns`) with the returns matrix.
        if len(labels) == len(returns):
            min_obs = max(50, returns.shape[1] * 5)
            cops: list[GaussianCopula | StudentTCopula] = []
            for k in range(2):
                mask = labels == k
                if mask.sum() < min_obs:
                    cops = []  # bail out, not enough data
                    break
                sub = returns.values[mask]
                if cfg.simulation.dependence == "student_t_copula":
                    cops.append(StudentTCopula.fit(sub))
                else:
                    cops.append(GaussianCopula.fit(sub))
            if len(cops) == 2:
                copula_by_regime = cops

    return FittedModels(
        tickers=tickers,
        marginals=marginals,
        garch=garch,
        copula=cop,
        regime=regime,
        copula_by_regime=copula_by_regime,
        mean=returns.mean().values,
    )


class MonteCarloEngine:
    """Generate simulated portfolio P&L paths from fitted models."""

    def __init__(self, cfg: EngineConfig, models: FittedModels) -> None:
        self.cfg = cfg
        self.models = models

    def run(self, weights: Sequence[float] | None = None) -> SimulationResult:
        cfg = self.cfg
        m = self.models
        n_paths = cfg.simulation.n_paths
        horizon = cfg.simulation.horizon_days
        n_assets = len(m.tickers)
        rng = np.random.default_rng(cfg.simulation.seed)

        w = np.array(weights if weights is not None else cfg.portfolio.weights, dtype=float)

        # 1. Per-path GARCH state. Each path carries its own conditional variance
        #    h_t and previous residual eps_{t-1}, initialized at the fitted last
        #    in-sample values, and evolves under the path's own shocks.
        if m.garch is not None:
            omega = np.array([g.omega for g in m.garch])
            alpha = np.array([g.alpha for g in m.garch])
            beta = np.array([g.beta for g in m.garch])
            asset_means = np.array([g.mean for g in m.garch])
            h_prev = np.tile(np.array([g.last_variance for g in m.garch]), (n_paths, 1))
            eps_prev = np.tile(np.array([g.last_resid for g in m.garch]), (n_paths, 1))
        else:
            omega = alpha = beta = None
            asset_means = m.mean
            sigmas = np.array([x.scale for x in m.marginals])
            h_prev = np.tile(sigmas**2, (n_paths, 1))
            eps_prev = np.zeros((n_paths, n_assets))

        # 2. Regime states (one Markov chain per path).
        if m.regime is not None:
            regime_states = np.stack(
                [m.regime.simulate_states(horizon, rng) for _ in range(n_paths)]
            )
            crisis = m.regime.crisis_state
            crisis_mult = float(m.regime.sigmas[crisis] / m.regime.sigmas[1 - crisis])
        else:
            regime_states = np.zeros((n_paths, horizon), dtype=np.int64)
            crisis = -1
            crisis_mult = 1.0

        # 3. Marginal standardization. We want unit-variance innovations going
        #    into the GARCH recursion, so normalize Student-t draws by their
        #    actual std (scale * sqrt(nu / (nu - 2)) for nu > 2).
        locs = np.array([mar.loc for mar in m.marginals])
        margin_stds = np.array(
            [
                mar.scale * np.sqrt(mar.df / (mar.df - 2.0)) if mar.df > 2 else mar.scale
                for mar in m.marginals
            ]
        )

        returns = np.empty((n_paths, horizon, n_assets))
        cond_vol = np.empty((n_paths, horizon, n_assets))
        for t in range(horizon):
            # GARCH update: h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}
            if omega is not None:
                h_t = omega + alpha * eps_prev**2 + beta * h_prev
            else:
                h_t = h_prev  # constant-vol fallback

            # Copula-coupled standardized shocks (zero mean, unit variance).
            # If regime-conditional copulas are available, draw each path from
            # the copula of its current regime — this models correlation
            # breakdown during stress (Student-t copula in crisis tends to have
            # higher correlation and stronger tail dependence).
            in_crisis_t = regime_states[:, t] == crisis                # (n_paths,)
            if m.copula_by_regime is not None and crisis >= 0:
                u = np.empty((n_paths, n_assets))
                for k, cop in enumerate(m.copula_by_regime):
                    mask = regime_states[:, t] == k
                    n_k = int(mask.sum())
                    if n_k:
                        u[mask] = cop.sample(n_k, rng)
            else:
                u = m.copula.sample(n_paths, rng)                     # (n_paths, n_assets)

            shocks = np.column_stack(
                [m.marginals[j].quantile(u[:, j]) for j in range(n_assets)]
            )
            z = (shocks - locs) / margin_stds                          # (n_paths, n_assets)
            if cfg.simulation.max_innovation_z is not None:
                cap = float(cfg.simulation.max_innovation_z)
                z = np.clip(z, -cap, cap)

            # Regime crisis amplifies conditional std; mean-shift could go here too.
            vol_mult = np.where(in_crisis_t, crisis_mult, 1.0)[:, None]  # (n_paths, 1)

            sigma_t = np.sqrt(h_t) * vol_mult                          # (n_paths, n_assets)
            eps_t = sigma_t * z
            r_t = asset_means + eps_t

            returns[:, t, :] = r_t
            cond_vol[:, t, :] = sigma_t
            # Feed eps_t (un-amplified for the GARCH recursion itself; the
            # crisis multiplier acts on the *realized* shock magnitude only).
            eps_prev = np.sqrt(h_t) * z
            h_prev = h_t

        cum = np.cumsum(returns, axis=1)  # log-return space
        portfolio_step_returns = returns @ w
        portfolio_terminal = portfolio_step_returns.sum(axis=1)
        return SimulationResult(
            asset_paths=cum,
            asset_returns=returns,
            portfolio_pnl=portfolio_terminal,
            weights=w,
            tickers=m.tickers,
        )


def _safe(_m: FittedModels) -> np.ndarray:
    """Fallback if no GARCH is fit — derive a flat covariance from marginal scales."""
    sigmas = np.array([x.scale for x in _m.marginals])
    return np.diag(sigmas)
