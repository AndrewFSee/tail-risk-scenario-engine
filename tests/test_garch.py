import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arch")

from tailrisk.models.garch import fit_garch


def test_garch_fit_and_simulate():
    rng = np.random.default_rng(0)
    # Simulate a simple ARCH-like series
    n = 1500
    eps = rng.standard_normal(n)
    sigma2 = np.empty(n)
    sigma2[0] = 1e-4
    r = np.empty(n)
    for t in range(1, n):
        sigma2[t] = 5e-6 + 0.1 * (r[t - 1] ** 2) + 0.85 * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * eps[t]
    fit = fit_garch(pd.Series(r))
    assert 0 < fit.alpha < 1
    assert 0 < fit.beta < 1
    sim_r, sim_h = fit.simulate_variance_path(50, rng)
    assert sim_r.shape == (50,)
    assert (sim_h > 0).all()
