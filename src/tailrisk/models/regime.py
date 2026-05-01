"""Two-state Markov regime-switching model for a return series.

State 0 = "normal" regime, state 1 = "crisis" regime (higher vol, lower mean).
Backed by ``statsmodels.tsa.regime_switching.MarkovRegression``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegimeFit:
    means: np.ndarray              # shape (2,)
    sigmas: np.ndarray             # shape (2,)
    transition: np.ndarray         # shape (2, 2), row-stochastic
    smoothed_probs: np.ndarray     # shape (T, 2)

    @property
    def crisis_state(self) -> int:
        """Index of the higher-volatility regime."""
        return int(np.argmax(self.sigmas))

    def stationary_distribution(self) -> np.ndarray:
        """Compute stationary distribution of the transition matrix."""
        eigvals, eigvecs = np.linalg.eig(self.transition.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        pi = np.real(eigvecs[:, idx])
        pi = pi / pi.sum()
        return pi

    def simulate_states(
        self,
        horizon: int,
        rng: np.random.Generator,
        start_state: int | None = None,
    ) -> np.ndarray:
        """Simulate a Markov chain of regime states of length ``horizon``."""
        if start_state is None:
            pi = self.stationary_distribution()
            start_state = int(rng.choice(2, p=pi))
        states = np.empty(horizon, dtype=np.int64)
        s = start_state
        for t in range(horizon):
            states[t] = s
            s = int(rng.choice(2, p=self.transition[s]))
        return states


def fit_markov_2state(returns: pd.Series | np.ndarray) -> RegimeFit:
    """Fit a 2-state Markov-switching model with switching mean and variance."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    r = pd.Series(np.asarray(returns)).dropna()
    model = MarkovRegression(r, k_regimes=2, trend="c", switching_variance=True)
    res = model.fit(disp=False)

    means = np.array([float(res.params[f"const[{i}]"]) for i in range(2)])
    sigmas = np.sqrt(
        np.array([float(res.params[f"sigma2[{i}]"]) for i in range(2)])
    )
    # statsmodels exposes the transition matrix via regime_transition (k x k x 1)
    # using the convention P[i, j] = P(S_t = i | S_{t-1} = j) — i.e. *column*
    # stochastic. We transpose to row-stochastic and renormalize defensively.
    P = np.asarray(res.regime_transition).squeeze()
    if P.shape != (2, 2):
        P = P.reshape(2, 2)
    P = P.T
    P = P / P.sum(axis=1, keepdims=True)
    smoothed = np.asarray(res.smoothed_marginal_probabilities)
    if smoothed.shape[0] == 2:
        smoothed = smoothed.T
    return RegimeFit(means=means, sigmas=sigmas, transition=P, smoothed_probs=smoothed)
