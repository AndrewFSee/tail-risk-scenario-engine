"""Rolling 1-day VaR backtest.

For tractability, the backtest fits a single GARCH(1,1)-t model to the
*portfolio* return series (rather than per-asset GARCH + copula at every
step), and forecasts the 1-day VaR analytically:

    VaR_alpha = -(mu + sigma_{t+1} * t_alpha(nu) * sqrt((nu - 2) / nu))

where the t-quantile is the alpha-quantile of the standardized Student-t.

The model is refit periodically on an expanding window, and between refits
we evolve the GARCH state forward using the realized residuals (recursive
out-of-sample forecasting, the standard academic protocol).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .coverage import CoverageReport, coverage_report


@dataclass
class BacktestResult:
    dates: pd.DatetimeIndex
    realized_returns: np.ndarray
    predicted_var_95: np.ndarray
    predicted_var_99: np.ndarray
    breaches_95: np.ndarray
    breaches_99: np.ndarray
    coverage_95: CoverageReport
    coverage_99: CoverageReport

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "realized": self.realized_returns,
                "var_95": self.predicted_var_95,
                "var_99": self.predicted_var_99,
                "breach_95": self.breaches_95,
                "breach_99": self.breaches_99,
            },
            index=self.dates,
        )


def _t_quantile_unit_variance(alpha: float, nu: float) -> float:
    """Lower-tail alpha-quantile of a Student-t standardized to unit variance."""
    q = stats.t.ppf(1 - alpha, df=nu)  # negative number
    if nu > 2:
        q = q * np.sqrt((nu - 2) / nu)
    return float(q)


def rolling_var_backtest(
    portfolio_returns: pd.Series,
    *,
    min_train: int = 500,
    refit_every: int = 63,
    alphas: tuple[float, ...] = (0.95, 0.99),
) -> BacktestResult:
    """Walk forward through ``portfolio_returns``, predicting each day's VaR.

    Parameters
    ----------
    portfolio_returns:
        Daily portfolio log returns indexed by date.
    min_train:
        Minimum number of observations before the first forecast (~2 trading
        years at 252).
    refit_every:
        Refit the GARCH model every K days; in between, just propagate the
        GARCH state recursion using realized residuals.
    alphas:
        Confidence levels to evaluate (we report breach stats for 95% and 99%).
    """
    from arch import arch_model

    r = portfolio_returns.dropna()
    if len(r) <= min_train + 10:
        raise ValueError(f"Not enough observations ({len(r)}) for backtest.")

    dates = r.index[min_train:]
    n_test = len(dates)
    realized = r.values[min_train:]
    var_curves = {a: np.empty(n_test) for a in alphas}

    # State carried between refits.
    omega = alpha_p = beta_p = nu = mu = None
    h_t: float | None = None
    eps_t: float | None = None

    for i in range(n_test):
        train_end = min_train + i
        if i == 0 or (i % refit_every == 0):
            # Refit on expanding window up to but not including the day we're
            # predicting.
            train = r.iloc[:train_end] * 100.0  # arch wants percent
            am = arch_model(train, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
            res = am.fit(disp="off")
            p = res.params
            omega = float(p["omega"]) / 1e4
            alpha_p = float(p["alpha[1]"])
            beta_p = float(p["beta[1]"])
            nu = float(p["nu"])
            mu = float(p["mu"]) / 100.0
            h_t = float(res.conditional_volatility.iloc[-1] ** 2) / 1e4
            eps_t = float(res.resid.iloc[-1]) / 100.0
        else:
            # Propagate state with the realized residual from t-1.
            assert omega is not None and alpha_p is not None and beta_p is not None
            r_prev = r.iloc[train_end - 1]
            eps_prev = r_prev - mu
            h_next = omega + alpha_p * eps_prev**2 + beta_p * h_t
            h_t = h_next
            eps_t = eps_prev

        # Forecast h_{t+1} for the day at index `train_end` (= dates[i]).
        h_forecast = omega + alpha_p * eps_t**2 + beta_p * h_t
        sigma_forecast = float(np.sqrt(h_forecast))
        for a in alphas:
            q = _t_quantile_unit_variance(a, nu)   # negative
            var_curves[a][i] = -(mu + sigma_forecast * q)

    var95 = var_curves[0.95]
    var99 = var_curves[0.99]
    cov95 = coverage_report(realized, var95, 0.95)
    cov99 = coverage_report(realized, var99, 0.99)
    return BacktestResult(
        dates=dates,
        realized_returns=realized,
        predicted_var_95=var95,
        predicted_var_99=var99,
        breaches_95=(-realized > var95).astype(int),
        breaches_99=(-realized > var99).astype(int),
        coverage_95=cov95,
        coverage_99=cov99,
    )
