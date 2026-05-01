"""Heavy-tailed marginal distributions for return modeling.

Two production-relevant choices are supported:

- Student-t: simple, captures excess kurtosis with a single tail parameter.
- Generalized Pareto Distribution (GPD): peaks-over-threshold EVT estimator
  for the *tail only*; the body of the distribution is left empirical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class StudentTFit:
    df: float
    loc: float
    scale: float

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return stats.t.rvs(self.df, loc=self.loc, scale=self.scale, size=size, random_state=rng)

    def quantile(self, q: float | np.ndarray) -> np.ndarray:
        return stats.t.ppf(q, self.df, loc=self.loc, scale=self.scale)


def fit_student_t(x: np.ndarray) -> StudentTFit:
    """MLE fit of a Student-t to a 1D return series."""
    df, loc, scale = stats.t.fit(np.asarray(x))
    return StudentTFit(df=float(df), loc=float(loc), scale=float(scale))


@dataclass
class GPDTailFit:
    """Peaks-over-threshold GPD fit on the *left* tail of a return series."""

    threshold: float       # negative value; losses below this are tail
    shape: float           # xi
    scale: float           # beta
    tail_prob: float       # P(X < threshold) from the empirical sample
    body: np.ndarray       # the non-tail observations (for hybrid sampling)

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        u = rng.random(size)
        out = np.empty(size)
        in_tail = u < self.tail_prob
        n_tail = int(in_tail.sum())
        if n_tail:
            # GPD exceedances are positive; we map back to the negative loss tail.
            exceed = stats.genpareto.rvs(
                self.shape, loc=0.0, scale=self.scale, size=n_tail, random_state=rng
            )
            out[in_tail] = self.threshold - exceed
        n_body = size - n_tail
        if n_body:
            out[~in_tail] = rng.choice(self.body, size=n_body, replace=True)
        return out


def fit_gpd_left_tail(x: np.ndarray, tail_quantile: float = 0.05) -> GPDTailFit:
    """Fit a GPD to the left-tail exceedances of ``x``.

    ``tail_quantile`` is the empirical probability mass treated as tail (e.g. 0.05).
    """
    x = np.asarray(x)
    threshold = float(np.quantile(x, tail_quantile))
    exceedances = threshold - x[x < threshold]  # positive magnitudes
    if exceedances.size < 20:
        raise ValueError(
            f"Too few tail observations ({exceedances.size}) to fit a GPD; "
            "increase the sample size or raise tail_quantile."
        )
    shape, _, scale = stats.genpareto.fit(exceedances, floc=0.0)
    body = x[x >= threshold]
    return GPDTailFit(
        threshold=threshold,
        shape=float(shape),
        scale=float(scale),
        tail_prob=float(tail_quantile),
        body=body,
    )
