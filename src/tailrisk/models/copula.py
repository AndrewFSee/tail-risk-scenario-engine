"""Copula-based dependence between assets.

We expose two copulas that matter for tail-risk work:

- Gaussian copula: standard but has zero asymptotic tail dependence.
- Student-t copula: produces symmetric tail dependence; the right choice
  when modeling correlation breakdowns during crises.

The implementations here are minimal and dependency-light — fit Pearson/Spearman
correlation and a degrees-of-freedom for the t-copula via profile MLE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _to_uniform(x: np.ndarray) -> np.ndarray:
    """Empirical CDF transform (per column) to (0, 1)."""
    n = x.shape[0]
    ranks = np.argsort(np.argsort(x, axis=0), axis=0) + 1
    return ranks / (n + 1.0)


@dataclass
class GaussianCopula:
    corr: np.ndarray   # (d, d)

    @classmethod
    def fit(cls, x: np.ndarray) -> "GaussianCopula":
        u = _to_uniform(x)
        z = stats.norm.ppf(u)
        return cls(corr=np.corrcoef(z, rowvar=False))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        d = self.corr.shape[0]
        L = np.linalg.cholesky(_nearest_psd(self.corr))
        z = rng.standard_normal(size=(n, d)) @ L.T
        return stats.norm.cdf(z)


@dataclass
class StudentTCopula:
    corr: np.ndarray
    df: float

    @classmethod
    def fit(cls, x: np.ndarray, df_grid: tuple[float, ...] = (3, 4, 5, 7, 10, 15, 25)) -> "StudentTCopula":
        u = _to_uniform(x)
        # Profile likelihood over df with corr from Gaussian-transformed data
        z = stats.norm.ppf(u)
        corr = np.corrcoef(z, rowvar=False)
        best_df, best_ll = float(df_grid[0]), -np.inf
        for nu in df_grid:
            ll = _t_copula_loglik(u, corr, nu)
            if ll > best_ll:
                best_ll, best_df = ll, float(nu)
        return cls(corr=corr, df=best_df)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        d = self.corr.shape[0]
        L = np.linalg.cholesky(_nearest_psd(self.corr))
        z = rng.standard_normal(size=(n, d)) @ L.T
        # Chi-squared scaling -> multivariate t
        s = rng.chisquare(self.df, size=n) / self.df
        t = z / np.sqrt(s)[:, None]
        return stats.t.cdf(t, df=self.df)


def _t_copula_loglik(u: np.ndarray, corr: np.ndarray, df: float) -> float:
    d = u.shape[1]
    t = stats.t.ppf(u, df=df)
    sign, logdet = np.linalg.slogdet(corr)
    inv = np.linalg.inv(_nearest_psd(corr))
    quad = np.einsum("ij,jk,ik->i", t, inv, t)
    # log density of multivariate t (unit scale matrix = corr)
    from scipy.special import gammaln
    log_num = gammaln((df + d) / 2.0) - gammaln(df / 2.0)
    log_den = (d / 2.0) * np.log(df * np.pi) + 0.5 * logdet
    log_mvt = log_num - log_den - ((df + d) / 2.0) * np.log1p(quad / df)
    log_marginals = stats.t.logpdf(t, df=df).sum(axis=1)
    return float((log_mvt - log_marginals).sum())


def _nearest_psd(a: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project to nearest symmetric PSD matrix (numerical hygiene)."""
    a = (a + a.T) / 2.0
    w, v = np.linalg.eigh(a)
    w = np.clip(w, eps, None)
    return (v * w) @ v.T
