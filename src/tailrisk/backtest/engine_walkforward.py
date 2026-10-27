"""Walk-forward validation of the full Monte Carlo engine at its own horizon.

``rolling_var_backtest`` checks a simpler stand-in (one GARCH on the portfolio series, 1-day
VaR). This checks what the engine actually reports: at each test date it refits every model
(per-asset GARCH, marginals, copulas, regimes) on data up to that date only, simulates the
next ``horizon`` days, and compares the predicted VaR with the realized portfolio return.
Test dates are ``horizon`` days apart, so outcomes don't overlap and the coverage tests hold.

It refits everything at every date, so it is slow (seconds per date); use fewer paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..config import EngineConfig
from .coverage import CoverageReport, coverage_report


@dataclass
class EngineBacktestResult:
    frame: pd.DataFrame            # per test date: predicted VaR/CVaR, realized return, breaches
    coverage_95: CoverageReport
    coverage_99: CoverageReport
    risk_tracking: float           # rank corr(predicted CVaR 95, realized |return|)


def engine_walkforward(
    returns: pd.DataFrame,
    cfg: EngineConfig,
    *,
    min_train: int = 1260,
    n_paths: int = 2000,
    progress=None,
) -> EngineBacktestResult:
    """Refit and simulate at every ``horizon``-th date after ``min_train`` observations."""
    from ..risk import tail_metrics
    from ..simulation import MonteCarloEngine, fit_models

    horizon = cfg.simulation.horizon_days
    w = np.asarray(cfg.portfolio.weights, dtype=float)
    c = cfg.model_copy(deep=True)
    c.simulation.n_paths = n_paths
    tests = list(range(min_train, len(returns) - horizon + 1, horizon))
    rows = []
    for k, t in enumerate(tests):
        models = fit_models(returns.iloc[:t], c)
        pnl = MonteCarloEngine(c, models).run().portfolio_pnl
        m95, m99 = tail_metrics(pnl, alpha=0.95), tail_metrics(pnl, alpha=0.99)
        realized = float(returns.iloc[t: t + horizon].values.sum(axis=0) @ w)
        rows.append({"date": returns.index[t], "var_95": m95.var, "cvar_95": m95.cvar, "var_99": m99.var,
                     "cvar_99": m99.cvar, "realized": realized})
        if progress:
            progress(k + 1, len(tests), rows[-1])
    df = pd.DataFrame(rows).set_index("date")
    df["breach_95"] = (-df["realized"] > df["var_95"]).astype(int)
    df["breach_99"] = (-df["realized"] > df["var_99"]).astype(int)
    tracking = float(stats.spearmanr(df["cvar_95"], df["realized"].abs())[0])
    return EngineBacktestResult(frame=df, coverage_95=coverage_report(df["realized"].values, df["var_95"].values, 0.95),
                                coverage_99=coverage_report(df["realized"].values, df["var_99"].values, 0.99),
                                risk_tracking=tracking)
