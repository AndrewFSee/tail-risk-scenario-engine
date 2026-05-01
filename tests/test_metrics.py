import numpy as np

from tailrisk.risk import conditional_var, max_drawdown, tail_metrics, value_at_risk


def test_var_and_cvar_ordering():
    rng = np.random.default_rng(0)
    pnl = rng.standard_t(df=5, size=20_000) * 0.01
    var = value_at_risk(pnl, 0.95)
    cvar = conditional_var(pnl, 0.95)
    assert cvar >= var > 0


def test_tail_metrics_fields():
    rng = np.random.default_rng(0)
    pnl = rng.standard_normal(5000) * 0.01
    m = tail_metrics(pnl, 0.99)
    assert m.alpha == 0.99
    assert m.var > 0


def test_max_drawdown_monotone_up_is_zero():
    cum = np.linspace(0, 0.1, 100)
    assert max_drawdown(cum) == 0.0
