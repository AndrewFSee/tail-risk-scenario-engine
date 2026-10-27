"""Hedging instrument payoff models.

Simplified, *engine-internal* models — not exchange-grade pricing. Intended
for *relative* comparison between hedge variants on simulated paths.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats


def black_scholes_put(spot, strike: float, years: float, vol: float, rate: float = 0.0):
    """European put value (per unit of underlying); intrinsic value once ``years`` <= 0."""
    spot = np.asarray(spot, dtype=float)
    if years <= 0 or vol <= 0:
        return np.maximum(strike - spot, 0.0)
    sd = vol * np.sqrt(years)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * years) / sd
    d2 = d1 - sd
    return strike * np.exp(-rate * years) * stats.norm.cdf(-d2) - spot * stats.norm.cdf(-d1)


@dataclass
class PutOption:
    """Long European put, bought today and marked to market at the simulation horizon.

    The underlying is identified by index ``asset_index`` in the simulated
    asset-return tensor. Strike is expressed as moneyness (K = moneyness * S0).
    ``tenor_days`` is calendar days (a "30-day put"); the horizon is trading days.

    The premium is ``premium_pct`` when given; otherwise Black-Scholes at
    ``implied_vol`` (annualized, e.g. the VIX / 100). A fixed premium ignores
    the market: a 1-month 5% OTM SPY put costs ~0.5% of notional at 20% vol
    but ~2% at 40%.
    """

    asset_index: int
    moneyness: float = 0.95
    premium_pct: float | None = None
    notional_fraction: float = 1.0    # notional as fraction of portfolio value
    tenor_days: int = 30
    implied_vol: float | None = None
    rate: float = 0.04

    def premium(self) -> float:
        if self.premium_pct is not None:
            return float(self.premium_pct)
        if self.implied_vol is None:
            raise ValueError("PutOption needs premium_pct or implied_vol")
        return float(black_scholes_put(1.0, self.moneyness, self.tenor_days / 365, self.implied_vol, self.rate))

    def payoff(self, asset_terminal_returns: np.ndarray, horizon_days: int | None = None) -> np.ndarray:
        """Net P&L per simulated path as a fraction of portfolio value: the put's value at the
        horizon (intrinsic if it has expired by then) minus the premium paid."""
        s_t_over_s0 = np.exp(asset_terminal_returns[:, self.asset_index])
        remaining = self.tenor_days / 365 - (horizon_days or 0) / 252
        if remaining > 1e-9 and self.implied_vol:
            value = black_scholes_put(s_t_over_s0, self.moneyness, remaining, self.implied_vol, self.rate)
        else:
            value = np.maximum(self.moneyness - s_t_over_s0, 0.0)
        return self.notional_fraction * (value - self.premium())


@dataclass
class InverseETF:
    """Static short-equity sleeve approximated as -1x the underlying return.

    A real -1x ETF has daily-rebalancing decay; we model that approximately
    by accumulating step-by-step rather than using terminal return.
    """

    asset_index: int
    allocation: float = 0.05

    def payoff_from_steps(self, asset_step_returns: np.ndarray) -> np.ndarray:
        """asset_step_returns: (n_paths, horizon, n_assets) log-returns."""
        simple = np.expm1(asset_step_returns[:, :, self.asset_index])
        # daily rebalanced -1x of the *simple* daily return: prod(1 - R_t) - 1
        compounded = np.prod(1.0 - simple, axis=1) - 1.0
        return self.allocation * compounded


def build_hedges(specs, tickers: list[str], implied_vol: float | None = None) -> list[PutOption | InverseETF]:
    """Hedge instruments from config specs (``HedgingConfig.instruments``), skipping any whose
    underlying is not in the portfolio. Puts without a fixed premium are priced at the spec's
    ``implied_vol``, else at ``implied_vol`` passed here (e.g. the current VIX / 100)."""
    idx = {t: i for i, t in enumerate(tickers)}
    out: list[PutOption | InverseETF] = []
    for spec in specs:
        if spec.kind == "put" and spec.underlying in idx:
            vol = spec.implied_vol or implied_vol
            if spec.premium_pct is None and vol is None:  # e.g. the VIX could not be fetched
                warnings.warn(f"put on {spec.underlying} skipped: no premium_pct and no implied vol to price it")
                continue
            out.append(PutOption(asset_index=idx[spec.underlying], moneyness=spec.moneyness or 0.95,
                                 premium_pct=spec.premium_pct, tenor_days=spec.tenor_days or 30,
                                 implied_vol=vol))
        elif spec.kind == "inverse_etf" and spec.ticker in idx:
            out.append(InverseETF(asset_index=idx[spec.ticker], allocation=spec.allocation or 0.05))
    return out
