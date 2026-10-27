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


def _garch_returns(n: int = 3000, seed: int = 5) -> pd.DataFrame:
    """Two assets with volatility clustering and moderately fat innovations (t, df 6)."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, (w, a, b) in {"A": (2e-6, 0.09, 0.89), "B": (1e-6, 0.06, 0.92)}.items():
        h, e, r = w / (1 - a - b), 0.0, []
        for _ in range(n):
            h = w + a * e**2 + b * h
            e = np.sqrt(h) * rng.standard_t(6) * np.sqrt(4 / 6)
            r.append(e)
        out[name] = r
    return pd.DataFrame(out)


def test_garch_marginals_fit_on_standardized_residuals():
    """Regression: marginals fitted on raw returns double-count the fat tails that GARCH
    already produces (df near 2 => infinite kurtosis and exploding simulated tails)."""
    from tailrisk.models import fit_student_t

    rets = _garch_returns()
    cfg = EngineConfig(
        portfolio=PortfolioConfig(tickers=["A", "B"], weights=[0.5, 0.5], start="2020-01-01"),
        simulation=SimulationConfig(marginal="student_t", volatility="garch", regime="none",
                                    dependence="gaussian"),
    )
    models = fit_models(rets, cfg)
    raw_df = fit_student_t(rets["A"].values).df
    assert models.marginals[0].df > raw_df + 1          # thinner than the raw-return fit
    assert 4 < models.marginals[0].df < 12               # near the true innovation df (6)
    assert models.garch[0].std_resid.shape == (len(rets),)


def test_regime_paths_start_from_given_probabilities():
    from tailrisk.models import RegimeFit

    r = RegimeFit(means=np.zeros(2), sigmas=np.array([0.01, 0.03]),
                  transition=np.array([[0.98, 0.02], [0.10, 0.90]]), smoothed_probs=np.array([[0.99, 0.01]]))
    rng = np.random.default_rng(0)
    calm = r.simulate_paths(20_000, 21, rng, start_probs=r.current_probs())
    assert calm[:, 0].mean() == pytest.approx(0.01, abs=0.005)        # today, not the long-run 1/6
    long_run = r.simulate_paths(20_000, 21, rng)
    assert long_run[:, 0].mean() == pytest.approx(r.stationary_distribution()[1], abs=0.01)
    # transitions follow the matrix
    prev, nxt = long_run[:, :-1].ravel(), long_run[:, 1:].ravel()
    assert nxt[prev == 0].mean() == pytest.approx(0.02, abs=0.003)
    assert nxt[prev == 1].mean() == pytest.approx(0.90, abs=0.01)


def test_regime_vol_boost_is_relative_to_today():
    """On a crisis day GARCH vol is already high; the regime must not stack its full crisis/calm
    ratio on top (walk-forward: crisis-start months had 95% VaR ~4.5x realized vol). Starting calm,
    moves into crisis still raise vol."""
    from tailrisk.models import RegimeFit

    rets = _garch_returns(n=1500)
    cfg = EngineConfig(
        portfolio=PortfolioConfig(tickers=["A", "B"], weights=[0.5, 0.5], start="2020-01-01"),
        simulation=SimulationConfig(n_paths=8000, horizon_days=21, seed=4, volatility="garch",
                                    regime="none", dependence="gaussian"),
    )
    models = fit_models(rets, cfg)
    base = MonteCarloEngine(cfg, models).run().portfolio_pnl.std()
    sticky = np.array([[0.95, 0.05], [0.05, 0.95]])
    for today, expect_boost in (([0.0, 1.0], False), ([1.0, 0.0], True)):
        models.regime = RegimeFit(means=np.zeros(2), sigmas=np.array([0.01, 0.03]), transition=sticky,
                                  smoothed_probs=np.array([today]))
        std = MonteCarloEngine(cfg, models).run().portfolio_pnl.std()
        if expect_boost:
            assert std > base * 1.1
        else:
            assert std == pytest.approx(base, rel=0.05)
