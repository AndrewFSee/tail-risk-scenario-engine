"""Price data loading and return computation."""

from __future__ import annotations

from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd


def load_prices(
    tickers: Sequence[str],
    start: date,
    end: date | None = None,
) -> pd.DataFrame:
    """Load adjusted close prices via yfinance.

    Returns a wide DataFrame indexed by date with one column per ticker.
    Imported lazily so the package is usable in offline / test contexts.
    """
    import yfinance as yf  # noqa: WPS433 (lazy import is intentional)

    df = yf.download(
        tickers=list(tickers),
        start=start.isoformat() if isinstance(start, date) else start,
        end=end.isoformat() if isinstance(end, date) else end,
        auto_adjust=True,
        progress=False,
    )
    # yfinance returns multi-index columns when len(tickers) > 1
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    else:
        df = df[["Close"]].rename(columns={"Close": tickers[0]})
    df = df.dropna(how="all").sort_index()
    return df[list(tickers)]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns from a price panel."""
    return np.log(prices / prices.shift(1)).dropna(how="any")


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="any")


def latest_vix() -> float | None:
    """Latest VIX close as an annualized decimal (e.g. 0.18), or None if unavailable.

    Used as the implied vol for pricing index puts: the VIX is a 30-day, skew-inclusive
    measure, so it sits near the implied vol of a ~5% out-of-the-money 1-month put."""
    try:
        import yfinance as yf  # noqa: WPS433

        closes = yf.download("^VIX", period="10d", auto_adjust=False, progress=False)["Close"].dropna()
        return float(np.asarray(closes).ravel()[-1]) / 100.0 if len(closes) else None
    except Exception:  # noqa: BLE001 - network or parsing problems: caller falls back
        return None
