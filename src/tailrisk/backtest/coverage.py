"""VaR backtesting and coverage tests.

Implements the two standard regulatory tests for VaR model validation:

- **Kupiec POF (Proportion of Failures, 1995)** — likelihood-ratio test for
  *unconditional coverage*: are the empirical breach rate and the model's
  nominal alpha consistent? Asymptotically chi-square(1) under H0.

- **Christoffersen (1998) independence test** — likelihood-ratio test for
  *independence* of breaches: under a well-calibrated model, breaches should
  not cluster. Asymptotically chi-square(1) under H0. Combined with Kupiec
  this gives the *conditional coverage* test (chi-square(2)).

Reference:
    Christoffersen, P. (1998), "Evaluating Interval Forecasts", IER 39(4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class CoverageReport:
    alpha: float
    n: int                 # number of forecasts evaluated
    breaches: int          # number of times realized loss > predicted VaR
    expected_breaches: float
    breach_rate: float

    # Kupiec (unconditional coverage)
    kupiec_lr: float
    kupiec_pvalue: float

    # Christoffersen (independence)
    christoffersen_lr: float
    christoffersen_pvalue: float

    # Combined (conditional coverage) = Kupiec + Christoffersen, chi2(2)
    cc_lr: float
    cc_pvalue: float

    def passes(self, level: float = 0.05) -> bool:
        """True iff we fail to reject both Kupiec and Christoffersen at ``level``."""
        return self.kupiec_pvalue > level and self.christoffersen_pvalue > level


def kupiec_test(breaches: np.ndarray, alpha: float) -> tuple[float, float]:
    """Likelihood-ratio test of unconditional coverage.

    H0: P(breach) == 1 - alpha.
    """
    breaches = np.asarray(breaches).astype(int)
    n = breaches.size
    x = int(breaches.sum())
    p_hat = x / n if n else 0.0
    p_null = 1.0 - alpha

    # Guard against log(0) when there are 0 or n breaches.
    eps = 1e-12
    p_hat_clip = min(max(p_hat, eps), 1 - eps)

    log_l_null = x * np.log(p_null) + (n - x) * np.log(1 - p_null)
    log_l_alt = x * np.log(p_hat_clip) + (n - x) * np.log(1 - p_hat_clip)
    lr = -2.0 * (log_l_null - log_l_alt)
    p_value = 1.0 - stats.chi2.cdf(lr, df=1)
    return float(lr), float(p_value)


def christoffersen_independence_test(breaches: np.ndarray) -> tuple[float, float]:
    """Likelihood-ratio test that breaches are independently distributed.

    H0: P(breach_t | breach_{t-1}) == P(breach_t | no breach_{t-1}).
    """
    b = np.asarray(breaches).astype(int)
    if b.size < 2:
        return 0.0, 1.0

    # Transition counts:
    #   n_ij = number of transitions from state i to state j
    n00 = n01 = n10 = n11 = 0
    for prev, curr in zip(b[:-1], b[1:]):
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    eps = 1e-12
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    pi01 = min(max(pi01, eps), 1 - eps)
    pi11 = min(max(pi11, eps), 1 - eps)
    pi = min(max(pi, eps), 1 - eps)

    log_l_null = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    log_l_alt = (
        n00 * np.log(1 - pi01) + n01 * np.log(pi01)
        + n10 * np.log(1 - pi11) + n11 * np.log(pi11)
    )
    lr = -2.0 * (log_l_null - log_l_alt)
    p_value = 1.0 - stats.chi2.cdf(lr, df=1)
    return float(lr), float(p_value)


def coverage_report(
    realized_returns: np.ndarray,
    predicted_var: np.ndarray,
    alpha: float,
) -> CoverageReport:
    """Run both Kupiec and Christoffersen on a series of VaR forecasts.

    A *breach* occurs when the realized loss exceeds the predicted VaR, i.e.
    ``-realized_returns > predicted_var``.
    """
    realized = np.asarray(realized_returns)
    var = np.asarray(predicted_var)
    if realized.shape != var.shape:
        raise ValueError("realized_returns and predicted_var must align in shape.")
    breaches = (-realized > var).astype(int)
    n = breaches.size
    expected = n * (1 - alpha)

    k_lr, k_p = kupiec_test(breaches, alpha)
    c_lr, c_p = christoffersen_independence_test(breaches)
    cc_lr = k_lr + c_lr
    cc_p = 1.0 - stats.chi2.cdf(cc_lr, df=2)

    return CoverageReport(
        alpha=alpha,
        n=n,
        breaches=int(breaches.sum()),
        expected_breaches=float(expected),
        breach_rate=float(breaches.mean()) if n else 0.0,
        kupiec_lr=k_lr,
        kupiec_pvalue=k_p,
        christoffersen_lr=c_lr,
        christoffersen_pvalue=c_p,
        cc_lr=float(cc_lr),
        cc_pvalue=float(cc_p),
    )
