"""LLM interpreter (stub).

Translate natural-language questions like:

    "What is my worst-case loss if tech crashes 20%?"

into a typed ``ScenarioRequest`` consumable by the engine, and execute it
against an already-fitted simulation to return concrete tail-risk numbers.

The default implementation is a regex-based stub; the OpenAI-backed version
is gated behind the ``llm`` extra. Keeping the interface explicit means we
can swap providers without touching the engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Map common natural-language sector words to ETF tickers we might hold.
_SECTOR_ALIASES = {
    "TECH": "QQQ",
    "TECHNOLOGY": "QQQ",
    "BOND": "TLT",
    "BONDS": "TLT",
    "TREASURIES": "TLT",
    "GOLD": "GLD",
    "ENERGY": "XLE",
    "STOCKS": "SPY",
    "EQUITIES": "SPY",
    "MARKET": "SPY",
}


@dataclass
class ScenarioRequest:
    kind: str                  # "shock_asset", "vol_shift", "historical_replay"
    asset: str | None = None
    magnitude_pct: float | None = None
    target: str | None = None  # for historical replay (e.g. "gfc_2008")


@dataclass
class ScenarioAnswer:
    request: ScenarioRequest
    summary: str
    metrics: dict[str, float] = field(default_factory=dict)
    pnl: np.ndarray | None = None


_SHOCK_RE = re.compile(
    r"(?P<asset>[A-Z]{1,5}|tech(?:nology)?|energy|bonds?|treasuries|gold|stocks|equities|market)\s+"
    r"(?:crash|drop|fall|tank|plung)\w*\s+"
    r"(?:by\s+)?(?P<pct>\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
_VOL_RE = re.compile(
    r"vol(?:atility)?\s+(?P<verb>doubl|tripl|halv|quadrupl)\w*",
    re.IGNORECASE,
)
_GFC_RE = re.compile(r"\b(2008|gfc|global\s+financial\s+crisis)\b", re.IGNORECASE)
_COVID_RE = re.compile(r"\b(covid|2020\s+crash|pandemic)\b", re.IGNORECASE)


def parse(query: str) -> ScenarioRequest:
    """Naive rule-based parser used as a fallback when no LLM is configured."""
    if _GFC_RE.search(query):
        return ScenarioRequest(kind="historical_replay", target="gfc_2008")
    if _COVID_RE.search(query):
        return ScenarioRequest(kind="historical_replay", target="covid_2020")
    m = _SHOCK_RE.search(query)
    if m:
        asset = m.group("asset").upper()
        asset = _SECTOR_ALIASES.get(asset, asset)
        return ScenarioRequest(
            kind="shock_asset",
            asset=asset,
            magnitude_pct=-float(m.group("pct")),
        )
    m = _VOL_RE.search(query)
    if m:
        verb = m.group("verb").lower()
        mult = {"doubl": 2.0, "tripl": 3.0, "quadrupl": 4.0, "halv": 0.5}[verb]
        return ScenarioRequest(kind="vol_shift", magnitude_pct=(mult - 1.0) * 100.0)
    raise ValueError(
        f"Could not interpret query: {query!r}. "
        "Try mentioning a ticker + 'crash N%', 'volatility doubles', or '2008'."
    )


def answer(
    query: str,
    sim: Any,
    *,
    historical_returns: pd.DataFrame | None = None,
    scenarios_dir: str | Path = "configs/scenarios",
    seed: int = 0,
) -> ScenarioAnswer:
    """Parse ``query`` and execute it against the simulation context.

    Parameters
    ----------
    query:
        Natural-language question.
    sim:
        A ``SimulationResult`` (has ``asset_returns``, ``weights``, ``tickers``).
    historical_returns:
        Required for ``historical_replay`` requests — a DataFrame of historical
        log returns indexed by date with columns matching ``sim.tickers``.
    scenarios_dir:
        Directory containing scenario YAML files (used for historical replay).
    """
    from ..risk.metrics import conditional_var, value_at_risk

    req = parse(query)
    weights = np.asarray(sim.weights)
    tickers = list(sim.tickers)
    asset_returns = np.asarray(sim.asset_returns)  # (n_paths, horizon, n_assets)

    if req.kind == "shock_asset":
        if req.asset not in tickers:
            return ScenarioAnswer(
                request=req,
                summary=(
                    f"'{req.asset}' is not in the portfolio "
                    f"(have {tickers}). Try one of those tickers."
                ),
            )
        idx = tickers.index(req.asset)
        # Apply shock as an additive log-return on the *terminal* horizon.
        shock = float(req.magnitude_pct) / 100.0  # e.g. -0.20
        terminal = asset_returns.sum(axis=1)
        terminal_shocked = terminal.copy()
        terminal_shocked[:, idx] = terminal_shocked[:, idx] + shock
        pnl = terminal_shocked @ weights
        metrics = _summarize_pnl(pnl)
        summary = (
            f"If {req.asset} drops an additional {abs(shock):.0%} on top of the "
            f"simulated horizon, expected portfolio return = {metrics['mean']:.2%}, "
            f"VaR 95% = {metrics['var_95']:.2%}, CVaR 95% = {metrics['cvar_95']:.2%}, "
            f"CVaR 99% = {metrics['cvar_99']:.2%}."
        )
        return ScenarioAnswer(request=req, summary=summary, metrics=metrics, pnl=pnl)

    if req.kind == "vol_shift":
        mult = 1.0 + float(req.magnitude_pct) / 100.0
        # Per-asset mean of terminal returns — scale dispersion around it.
        terminal = asset_returns.sum(axis=1)
        mean = terminal.mean(axis=0, keepdims=True)
        terminal_scaled = mean + (terminal - mean) * mult
        pnl = terminal_scaled @ weights
        metrics = _summarize_pnl(pnl)
        summary = (
            f"With volatility scaled by {mult:.2f}x: "
            f"expected portfolio return = {metrics['mean']:.2%}, "
            f"VaR 95% = {metrics['var_95']:.2%}, CVaR 95% = {metrics['cvar_95']:.2%}, "
            f"CVaR 99% = {metrics['cvar_99']:.2%}."
        )
        return ScenarioAnswer(request=req, summary=summary, metrics=metrics, pnl=pnl)

    if req.kind == "historical_replay":
        if historical_returns is None:
            return ScenarioAnswer(
                request=req,
                summary="Historical replay requires the historical returns "
                "DataFrame; pass historical_returns=...",
            )
        from ..stress import HistoricalScenario, load_scenario

        scen_path = Path(scenarios_dir) / f"{req.target}.yaml"
        if not scen_path.exists():
            return ScenarioAnswer(
                request=req,
                summary=f"No scenario file found at {scen_path}.",
            )
        scen = load_scenario(scen_path)
        if not isinstance(scen, HistoricalScenario):
            return ScenarioAnswer(
                request=req, summary=f"{req.target} is not a historical scenario."
            )
        # Align columns to portfolio tickers.
        cols = [c for c in tickers if c in historical_returns.columns]
        rng = np.random.default_rng(seed)
        paths = scen.sample_paths(
            historical_returns[cols],
            n_paths=asset_returns.shape[0],
            horizon=asset_returns.shape[1],
            rng=rng,
        )
        # If some tickers are missing from the historical window, zero-fill.
        if cols != tickers:
            full = np.zeros((paths.shape[0], paths.shape[1], len(tickers)))
            for j, c in enumerate(cols):
                full[:, :, tickers.index(c)] = paths[:, :, j]
            paths = full
        terminal = paths.sum(axis=1)
        pnl = terminal @ weights
        metrics = _summarize_pnl(pnl)
        summary = (
            f"Replaying scenario '{req.target}' "
            f"({scen.start} to {scen.end}, block-bootstrap size {scen.block_size}): "
            f"expected portfolio return = {metrics['mean']:.2%}, "
            f"VaR 95% = {metrics['var_95']:.2%}, CVaR 95% = {metrics['cvar_95']:.2%}, "
            f"CVaR 99% = {metrics['cvar_99']:.2%}."
        )
        return ScenarioAnswer(request=req, summary=summary, metrics=metrics, pnl=pnl)

    return ScenarioAnswer(request=req, summary=f"Unhandled scenario kind: {req.kind}")


def _summarize_pnl(pnl: np.ndarray) -> dict[str, float]:
    from ..risk.metrics import conditional_var, value_at_risk

    return {
        "mean": float(np.mean(pnl)),
        "std": float(np.std(pnl)),
        "var_95": value_at_risk(pnl, 0.95),
        "cvar_95": conditional_var(pnl, 0.95),
        "var_99": value_at_risk(pnl, 0.99),
        "cvar_99": conditional_var(pnl, 0.99),
    }
