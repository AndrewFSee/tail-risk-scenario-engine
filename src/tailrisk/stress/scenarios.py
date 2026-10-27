"""Stress test scenarios.

Two scenario families:

- *Historical*: replay or block-bootstrap a historical window (e.g. GFC 2008,
  COVID-2020). The output is a sample of return paths drawn from that window.
- *Parametric*: apply named shocks (vol multiplier, mean shift, correlation
  floor) to the fitted models before simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml


@dataclass
class HistoricalScenario:
    name: str
    description: str
    start: str
    end: str
    block_size: int = 5

    def sample_paths(
        self,
        returns: pd.DataFrame,
        n_paths: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Block-bootstrap returns from the historical window.

        Returns a tensor of shape (n_paths, horizon, n_assets) of log returns.
        """
        window = returns.loc[self.start : self.end].values
        T, N = window.shape
        if T < self.block_size:
            raise ValueError(f"Historical window {self.name!r} too short ({T} rows).")

        out = np.empty((n_paths, horizon, N))
        for p in range(n_paths):
            buf = []
            while len(buf) < horizon:
                start = int(rng.integers(0, T - self.block_size + 1))
                buf.extend(window[start : start + self.block_size].tolist())
            out[p] = np.asarray(buf[:horizon])
        return out


@dataclass
class ParametricScenario:
    name: str
    description: str
    vol_multiplier: float = 1.0
    mean_shift_bps: float = 0.0
    correlation_floor: float | None = None


Scenario = HistoricalScenario | ParametricScenario


def load_scenario(path: str | Path) -> Scenario:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    kind: Literal["historical", "parametric"] = data["type"]
    if kind == "historical":
        return HistoricalScenario(
            name=data["name"],
            description=data.get("description", ""),
            start=data["window"]["start"],
            end=data["window"]["end"],
            block_size=int(data.get("block_size", 5)),
        )
    if kind == "parametric":
        s = data.get("shocks", {})
        return ParametricScenario(
            name=data["name"],
            description=data.get("description", ""),
            vol_multiplier=float(s.get("vol_multiplier", 1.0)),
            mean_shift_bps=float(s.get("mean_shift_bps", 0.0)),
            correlation_floor=s.get("correlation_floor"),
        )
    raise ValueError(f"Unknown scenario type: {kind!r}")


def _floor_correlation(corr: np.ndarray, floor: float) -> np.ndarray:
    """Raise every off-diagonal correlation to at least ``floor``, then repair to a valid
    correlation matrix (nearest PSD, unit diagonal)."""
    from ..models.copula import _nearest_psd

    out = np.array(corr, dtype=float, copy=True)
    off = ~np.eye(out.shape[0], dtype=bool)
    out[off] = np.maximum(out[off], floor)
    out = _nearest_psd(out)
    d = np.sqrt(np.diag(out))
    return out / np.outer(d, d)


def apply_parametric(models, scenario: ParametricScenario):
    """A copy of fitted models with the scenario's shocks applied.

    - ``vol_multiplier`` k scales volatility by k: with GARCH, omega and the starting variance by
      k^2 and the last residual by k, so the whole variance path scales by k^2; without GARCH,
      the marginal scales by k.
    - ``mean_shift_bps`` is added to every asset's daily drift.
    - ``correlation_floor`` raises pairwise copula correlations (all regimes) to at least that.
    """
    import copy

    m = copy.deepcopy(models)
    k = float(scenario.vol_multiplier)
    if k != 1.0:
        if m.garch is not None:
            for g in m.garch:
                g.omega *= k**2
                g.last_variance *= k**2
                g.last_resid *= k
        else:
            for mar in m.marginals:
                mar.scale *= k
    if scenario.mean_shift_bps:
        m.mean = np.asarray(m.mean, dtype=float) + scenario.mean_shift_bps / 1e4
    if scenario.correlation_floor is not None:
        f = float(scenario.correlation_floor)
        m.copula.corr = _floor_correlation(m.copula.corr, f)
        for cop in m.copula_by_regime or []:
            cop.corr = _floor_correlation(cop.corr, f)
    return m


def run_scenario(scenario: Scenario, returns: pd.DataFrame, models, cfg) -> np.ndarray:
    """Simulated ``horizon``-day portfolio log returns under a scenario, shape (n_paths,).

    Historical scenarios block-bootstrap their window; parametric ones re-run the Monte Carlo
    engine on shocked models (same seed as the baseline, so differences are the shock)."""
    from ..simulation import MonteCarloEngine

    w = np.asarray(cfg.portfolio.weights, dtype=float)
    if isinstance(scenario, HistoricalScenario):
        rng = np.random.default_rng(cfg.simulation.seed)
        paths = scenario.sample_paths(returns[models.tickers], cfg.simulation.n_paths,
                                      cfg.simulation.horizon_days, rng)
        return paths.sum(axis=1) @ w
    return MonteCarloEngine(cfg, apply_parametric(models, scenario)).run().portfolio_pnl
