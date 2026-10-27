"""Publish a tail-risk run into the Dashboard Hub (market-compass).

The hub's ``run_jobs.py`` runs this with *this* project's venv: it sets
``HUB_DATA_DIR`` and puts the hub folder on ``PYTHONPATH`` so
``from core import store`` resolves to the hub's storage module.

    python scripts/publish_to_hub.py                       # configs/default.yaml
    python scripts/publish_to_hub.py -c configs/x.yaml

One simulation (the CLI's ``simulate``, ``hedge`` and ``optimize`` each run their
own with the same seed) feeds every output, written to ``<HUB_DATA_DIR>/tailrisk/``:

* ``fan``       portfolio cumulative return percentiles by day (5/25/50/75/95)
* ``pnl_hist``  histogram of terminal P&L, unhedged and with each hedge
* ``summary``   (JSON) tail metrics, the same metrics from history, hedge
                comparison (puts priced at the current VIX), stress scenarios,
                CVaR-optimal weights, current crisis-regime probability
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tailrisk.config import EngineConfig  # noqa: E402
from tailrisk.data import load_prices, log_returns  # noqa: E402
from tailrisk.data import latest_vix  # noqa: E402
from tailrisk.hedging import build_hedges, evaluate_hedges  # noqa: E402
from tailrisk.stress import load_scenario, run_scenario  # noqa: E402
from tailrisk.optimization import optimize_cvar  # noqa: E402
from tailrisk.risk import tail_metrics  # noqa: E402
from tailrisk.simulation import MonteCarloEngine, fit_models  # noqa: E402

import yfinance  # noqa: E402,F401  (imported here so the filter below outlives its own)

# yfinance still uses pandas APIs deprecated in pandas 3, and on import it re-enables
# those warnings with its own filter - so this has to come after the import
warnings.filterwarnings("ignore", message="The inplace keyword")

try:
    from core import store  # the hub's storage module (hub folder is on PYTHONPATH)
except ImportError:
    sys.exit("Could not import the hub's core.store: run this through market-compass's run_jobs.py, "
             "or put the market-compass folder on PYTHONPATH.")

MODULE = "tailrisk"
FAN_PCTS = [5, 25, 50, 75, 95]


def log(msg: str) -> None:
    print(msg, flush=True)


def _metrics(pnl: np.ndarray, levels) -> dict:
    out = {}
    for a in levels:
        m = tail_metrics(pnl, alpha=a)
        out[f"{a:.0%}"] = {"var": m.var, "cvar": m.cvar, "mean": m.mean, "std": m.std,
                           "excess_kurtosis": m.excess_kurtosis}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--demo", action="store_true", help="fewer paths, for a quick check (still downloads prices)")
    args = parser.parse_args()

    cfg = EngineConfig.from_yaml(args.config)
    if args.demo:
        cfg.simulation.n_paths = 2000
    t = time.time()
    rets = log_returns(load_prices(cfg.portfolio.tickers, cfg.portfolio.start, cfg.portfolio.end))
    log(f"Loaded {len(rets)} daily returns for {', '.join(rets.columns)} "
        f"({rets.index[0]:%Y-%m-%d} .. {rets.index[-1]:%Y-%m-%d})")
    models = fit_models(rets, cfg)
    sim = MonteCarloEngine(cfg, models).run()
    log(f"Simulated {cfg.simulation.n_paths} paths x {cfg.simulation.horizon_days} days in {time.time() - t:.0f}s")
    w, horizon, levels = sim.weights, cfg.simulation.horizon_days, cfg.risk.confidence_levels

    # -- what history says, on the same definition (sum of weighted daily log returns)
    hist = (rets.to_numpy() @ w)
    hist_h = pd.Series(hist).rolling(horizon).sum().dropna().to_numpy()

    # -- hedges and CVaR-optimal weights on the same scenarios -----------
    vix = latest_vix()
    hedges = evaluate_hedges(sim.asset_returns, w, build_hedges(cfg.hedging.instruments, sim.tickers,
                                                                implied_vol=vix))

    # -- stress scenarios (configs/scenarios/*.yaml) on the same fitted models
    stress = [{"name": "baseline", "description": "fitted models, no shock", **_metrics(sim.portfolio_pnl, levels)}]
    for path in sorted((PROJECT_ROOT / "configs" / "scenarios").glob("*.yaml")):
        scen = load_scenario(path)
        stress.append({"name": scen.name, "description": scen.description,
                       **_metrics(run_scenario(scen, rets, models, cfg), levels)})
    opt = None
    if cfg.optimization.enabled:
        res = optimize_cvar(sim.asset_returns.sum(axis=1), alpha=cfg.optimization.cvar_alpha,
                            long_only=cfg.optimization.long_only, max_weight=cfg.optimization.max_weight)
        opt = {"alpha": cfg.optimization.cvar_alpha, "weights": dict(zip(sim.tickers, map(float, res.weights))),
               "cvar": float(res.cvar), "var": float(res.var), "expected_return": float(res.expected_return)}

    crisis = None
    if models.regime is not None:
        r = models.regime
        crisis = {"prob": float(r.smoothed_probs[-1, r.crisis_state]),
                  "long_run_share": float(r.stationary_distribution()[r.crisis_state]),
                  "vol_ratio": float(r.sigmas.max() / r.sigmas.min())}

    # -- tables -----------------------------------------------------------
    port_paths = np.cumsum(sim.asset_returns @ w, axis=1)  # (n_paths, horizon)
    fan = pd.DataFrame(np.percentile(port_paths, FAN_PCTS, axis=0).T, columns=[f"p{p}" for p in FAN_PCTS],
                       index=pd.RangeIndex(1, horizon + 1, name="day"))
    edges = np.linspace(np.percentile(sim.portfolio_pnl, 0.1), np.percentile(sim.portfolio_pnl, 99.9), 81)
    hist_rows = {"bin_mid": (edges[:-1] + edges[1:]) / 2}
    hist_rows["unhedged"] = np.histogram(sim.portfolio_pnl, edges)[0] / len(sim.portfolio_pnl)
    pnl_hist = pd.DataFrame(hist_rows)

    summary = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "demo": args.demo,
        "data_through": rets.index[-1].strftime("%Y-%m-%d"),
        "data_start": rets.index[0].strftime("%Y-%m-%d"),
        "portfolio": dict(zip(sim.tickers, map(float, w))),
        "model": {"n_paths": cfg.simulation.n_paths, "horizon_days": horizon, "marginal": cfg.simulation.marginal,
                  "volatility": cfg.simulation.volatility, "regime": cfg.simulation.regime,
                  "dependence": cfg.simulation.dependence},
        "simulated": _metrics(sim.portfolio_pnl, levels),
        "historical": _metrics(hist_h, levels),
        "hedges": [{"name": h.name, "expected_return": h.expected_return, "var_95": h.var_95,
                    "cvar_95": h.cvar_95, "cvar_99": h.cvar_99} for h in hedges],
        "optimal": opt,
        "hedge_implied_vol": vix,
        "stress": stress,
        "crisis_regime": crisis,
    }
    store.save_frame(MODULE, "fan", fan.astype(np.float64))
    store.save_frame(MODULE, "pnl_hist", pnl_hist.astype(np.float64))
    store.save_json(MODULE, "summary", summary)
    s95 = summary["simulated"]["95%"]
    log(f"Published tail risk: {horizon}d VaR95 {s95['var']:.2%}, CVaR95 {s95['cvar']:.2%}, "
        f"E[R] {s95['mean']:+.2%}" + (f", P(crisis) {crisis['prob']:.0%}" if crisis else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
