import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arch")

from tailrisk.config import EngineConfig, PortfolioConfig, SimulationConfig
from tailrisk.simulation import MonteCarloEngine, fit_models
from tailrisk.stress import HistoricalScenario, ParametricScenario, apply_parametric, run_scenario


def _setup(volatility="garch"):
    rng = np.random.default_rng(1)
    common = rng.standard_t(5, size=1500) * 0.006
    rets = pd.DataFrame({"A": common + rng.standard_t(5, size=1500) * 0.006,
                         "B": 0.3 * common + rng.standard_t(5, size=1500) * 0.006},
                        index=pd.bdate_range("2015-01-01", periods=1500))
    cfg = EngineConfig(portfolio=PortfolioConfig(tickers=["A", "B"], weights=[0.5, 0.5], start="2015-01-01"),
                       simulation=SimulationConfig(n_paths=4000, horizon_days=10, seed=3, volatility=volatility,
                                                   regime="none", dependence="gaussian"))
    return rets, cfg, fit_models(rets, cfg)


@pytest.mark.parametrize("volatility", ["garch", "constant"])
def test_vol_multiplier_scales_simulated_volatility(volatility):
    rets, cfg, models = _setup(volatility)
    base = MonteCarloEngine(cfg, models).run().portfolio_pnl
    doubled = run_scenario(ParametricScenario("x2", "", vol_multiplier=2.0), rets, models, cfg)
    assert doubled.std() / base.std() == pytest.approx(2.0, rel=0.1)


def test_mean_shift_and_correlation_floor():
    rets, cfg, models = _setup()
    shocked = apply_parametric(models, ParametricScenario("s", "", mean_shift_bps=-10, correlation_floor=0.8))
    assert np.allclose(shocked.mean, models.mean - 0.001)
    assert shocked.copula.corr[0, 1] >= 0.8 - 1e-9 and np.allclose(np.diag(shocked.copula.corr), 1.0)
    assert models.copula.corr[0, 1] < 0.8                     # the original models are untouched


def test_historical_scenario_bootstraps_its_window():
    rets, cfg, models = _setup()
    window = HistoricalScenario("w", "", start="2016-01-01", end="2016-06-30")
    pnl = run_scenario(window, rets, models, cfg)
    assert pnl.shape == (cfg.simulation.n_paths,)
    w_daily = rets.loc["2016-01-01":"2016-06-30"].values @ np.array([0.5, 0.5])
    assert pnl.mean() == pytest.approx(w_daily.mean() * cfg.simulation.horizon_days, abs=0.002)
