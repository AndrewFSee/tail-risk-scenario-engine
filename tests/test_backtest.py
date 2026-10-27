import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arch")

from tailrisk.backtest import (
    christoffersen_independence_test,
    coverage_report,
    kupiec_test,
    rolling_var_backtest,
)


def test_kupiec_accepts_well_calibrated_breaches():
    rng = np.random.default_rng(0)
    n = 5000
    breaches = (rng.random(n) < 0.05).astype(int)  # exactly 5% expected
    lr, p = kupiec_test(breaches, alpha=0.95)
    assert p > 0.05  # fail to reject H0 of correct coverage


def test_kupiec_rejects_too_many_breaches():
    rng = np.random.default_rng(1)
    n = 5000
    breaches = (rng.random(n) < 0.20).astype(int)  # 20% breach rate vs nominal 5%
    lr, p = kupiec_test(breaches, alpha=0.95)
    assert p < 1e-6


def test_christoffersen_detects_clustered_breaches():
    # Construct a sequence where breaches cluster (a breach makes another breach
    # very likely on the next step) — independence should be rejected.
    rng = np.random.default_rng(2)
    n = 4000
    b = np.zeros(n, dtype=int)
    state = 0
    for t in range(n):
        if state == 0:
            state = 1 if rng.random() < 0.03 else 0
        else:
            state = 1 if rng.random() < 0.6 else 0
        b[t] = state
    lr, p = christoffersen_independence_test(b)
    assert p < 0.01


def test_christoffersen_accepts_independent_breaches():
    rng = np.random.default_rng(3)
    b = (rng.random(5000) < 0.05).astype(int)
    lr, p = christoffersen_independence_test(b)
    assert p > 0.05


def test_coverage_report_fields():
    rng = np.random.default_rng(4)
    realized = rng.standard_normal(2000) * 0.01
    var = np.full_like(realized, 0.03)  # very loose VaR -> few breaches
    cov = coverage_report(realized, var, alpha=0.95)
    assert cov.n == 2000
    assert cov.breaches < cov.expected_breaches  # too few breaches
    assert 0.0 <= cov.kupiec_pvalue <= 1.0
    assert 0.0 <= cov.christoffersen_pvalue <= 1.0


def test_rolling_var_backtest_runs():
    rng = np.random.default_rng(5)
    n = 800
    # Simple GARCH-like series so the model has something to fit.
    eps = rng.standard_t(df=6, size=n) * 0.01
    r = pd.Series(eps, index=pd.bdate_range("2018-01-01", periods=n))
    res = rolling_var_backtest(r, min_train=400, refit_every=63)
    assert res.predicted_var_95.shape[0] == n - 400
    assert (res.predicted_var_95 > 0).all()
    assert (res.predicted_var_99 >= res.predicted_var_95).all()
    assert 0 <= res.coverage_95.breaches <= res.coverage_95.n


def test_propagation_matches_arch_filter():
    """Between refits the GARCH state must follow the standard recursion: with one fit and
    only propagation afterwards, the forecasts equal arch's conditional volatility for the
    same fixed parameters. (A previous version fed each shock in twice: errors up to ~23%.)"""
    import pandas as pd
    from arch import arch_model

    from tailrisk.backtest.rolling import _t_quantile_unit_variance, rolling_var_backtest

    rng = np.random.default_rng(0)
    n, train = 1200, 500
    w, a, b = 2e-6, 0.08, 0.9
    h, e, r = w / (1 - a - b), 0.0, []
    for _ in range(n):
        h = w + a * e**2 + b * h
        e = np.sqrt(h) * rng.standard_t(6) * np.sqrt(4 / 6)
        r.append(0.0003 + e)
    r = pd.Series(r, index=pd.bdate_range("2015-01-01", periods=n))

    bt = rolling_var_backtest(r, min_train=train, refit_every=10_000)
    fit = arch_model(r.iloc[:train] * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    sigma_true = arch_model(r * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fix(
        fit.params).conditional_volatility.values[train:] / 100
    q = _t_quantile_unit_variance(0.95, fit.params["nu"])
    sigma_bt = (-bt.predicted_var_95 - fit.params["mu"] / 100) / q
    np.testing.assert_allclose(sigma_bt, sigma_true, rtol=1e-6)


def test_engine_walkforward_on_synthetic_data():
    import pandas as pd

    from tailrisk.backtest import engine_walkforward
    from tailrisk.config import EngineConfig, PortfolioConfig, SimulationConfig

    rng = np.random.default_rng(2)
    rets = pd.DataFrame(rng.standard_t(6, size=(640, 2)) * 0.008, columns=["A", "B"],
                        index=pd.bdate_range("2018-01-01", periods=640))
    cfg = EngineConfig(portfolio=PortfolioConfig(tickers=["A", "B"], weights=[0.6, 0.4], start="2018-01-01"),
                       simulation=SimulationConfig(horizon_days=21, volatility="garch", regime="none",
                                                   dependence="gaussian"))
    res = engine_walkforward(rets, cfg, min_train=500, n_paths=500)
    assert len(res.frame) == 6                                   # (640 - 21 - 500) // 21 + 1
    assert (res.frame["var_99"] > res.frame["var_95"]).all()
    first = res.frame.index[0]
    assert first == rets.index[500]                              # first test uses only the first 500 days
    assert res.frame.loc[first, "realized"] == pytest.approx(rets.iloc[500:521].values.sum(axis=0) @ [0.6, 0.4])
