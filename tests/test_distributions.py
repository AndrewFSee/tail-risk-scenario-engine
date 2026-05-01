import numpy as np
import pytest

from tailrisk.models.distributions import fit_gpd_left_tail, fit_student_t


def test_student_t_fit_recovers_scale():
    rng = np.random.default_rng(0)
    x = rng.standard_t(df=5, size=5000) * 0.02
    fit = fit_student_t(x)
    assert 2.5 < fit.df < 12
    assert fit.scale == pytest.approx(0.02, rel=0.2)


def test_gpd_left_tail_samples_negative():
    rng = np.random.default_rng(0)
    x = rng.standard_t(df=4, size=5000) * 0.015
    fit = fit_gpd_left_tail(x, tail_quantile=0.05)
    s = fit.sample(2000, rng)
    # Some samples should fall below the threshold.
    assert (s < fit.threshold).any()
    assert s.shape == (2000,)
