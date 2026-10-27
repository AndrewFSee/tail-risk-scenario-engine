import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arch")
pytest.importorskip("statsmodels")

from tailrisk.config import EngineConfig, PortfolioConfig, SimulationConfig
from tailrisk.simulation import MonteCarloEngine, fit_models


def _synthetic_returns(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.standard_t(df=5, size=n) * 0.012
    b = rng.standard_t(df=6, size=n) * 0.009
    return pd.DataFrame({"A": a, "B": b})


def test_engine_runs_end_to_end():
    rets = _synthetic_returns()
    cfg = EngineConfig(
        portfolio=PortfolioConfig(
            tickers=["A", "B"], weights=[0.5, 0.5], start="2020-01-01"
        ),
        simulation=SimulationConfig(
            n_paths=500, horizon_days=10, seed=1,
            marginal="student_t", volatility="garch",
            regime="markov_2state", dependence="student_t_copula",
        ),
    )
    models = fit_models(rets, cfg)
    sim = MonteCarloEngine(cfg, models).run()
    assert sim.portfolio_pnl.shape == (500,)
    assert sim.asset_returns.shape == (500, 10, 2)
    # Sanity: mean is finite, std is positive.
    assert np.isfinite(sim.portfolio_pnl.mean())
    assert sim.portfolio_pnl.std() > 0


def test_per_path_garch_produces_path_specific_volatility():
    """Per-path GARCH should give different conditional vol across paths,
    not a single shared vol trajectory."""
    rets = _synthetic_returns(n=1500, seed=2)
    cfg = EngineConfig(
        portfolio=PortfolioConfig(
            tickers=["A", "B"], weights=[0.5, 0.5], start="2020-01-01"
        ),
        simulation=SimulationConfig(
            n_paths=400, horizon_days=15, seed=3,
            marginal="student_t", volatility="garch",
            regime="none", dependence="gaussian",
        ),
    )
    models = fit_models(rets, cfg)
    sim = MonteCarloEngine(cfg, models).run()
    # For asset A, take per-path realized std across the 15-day horizon.
    per_path_vol = sim.asset_returns[:, :, 0].std(axis=1)
    spread = per_path_vol.max() - per_path_vol.min()
    median_vol = np.median(per_path_vol)
    # If vol were shared across paths, spread would be ~0 (only sampling noise).
    # Per-path GARCH should give spread comparable to or larger than the median.
    assert spread > median_vol, (
        f"Expected per-path vol dispersion; got spread={spread:.4f} "
        f"vs median={median_vol:.4f}"
    )


def test_garch_drift_is_sample_mean_not_t_location():
    """Regression: GARCH-t ``mu`` is a robust (median-like) location; for skewed returns
    it differs from the mean, and using it as drift inflated simulated returns ~2.5x."""
    from tailrisk.models.garch import fit_garch

    rng = np.random.default_rng(3)
    n = 3000
    # mostly small gains, occasional large losses: location > mean
    r = rng.standard_t(df=5, size=n) * 0.008 + 0.001 - (rng.random(n) < 0.03) * 0.04
    rets = pd.DataFrame({"A": r})
    horizon = 10
    assert fit_garch(rets["A"]).mean - rets["A"].mean() > 0.0008  # the gap this test is about
    cfg = EngineConfig(
        portfolio=PortfolioConfig(tickers=["A"], weights=[1.0], start="2020-01-01"),
        simulation=SimulationConfig(n_paths=4000, horizon_days=horizon, seed=2, marginal="student_t",
                                    volatility="garch", regime="none", dependence="gaussian"),
    )
    sim = MonteCarloEngine(cfg, fit_models(rets, cfg)).run()
    simulated = sim.asset_returns.sum(axis=1).mean()
    assert simulated == pytest.approx(horizon * rets["A"].mean(), abs=0.004)
