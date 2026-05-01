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
