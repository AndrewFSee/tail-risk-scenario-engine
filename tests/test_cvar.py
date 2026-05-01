import numpy as np
import pytest

pytest.importorskip("cvxpy")

from tailrisk.optimization import optimize_cvar


def test_optimize_cvar_long_only_simplex():
    rng = np.random.default_rng(0)
    # Asset 0 is risky with fat left tail, asset 1 is mild.
    n = 5000
    a = rng.standard_t(df=3, size=n) * 0.02 + 0.0005
    b = rng.standard_normal(n) * 0.005 + 0.0002
    R = np.column_stack([a, b])
    res = optimize_cvar(R, alpha=0.95, long_only=True, max_weight=1.0)
    assert res.weights.shape == (2,)
    assert pytest.approx(res.weights.sum(), abs=1e-6) == 1.0
    assert (res.weights >= -1e-9).all()
    # The optimizer should prefer the lower-tail-risk asset.
    assert res.weights[1] > res.weights[0]
