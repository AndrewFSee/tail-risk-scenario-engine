"""Generate recruiter-friendly PNGs for the README.

Run:
    python scripts/generate_readme_figures.py

Produces in docs/images/:
    fan_chart.png
    loss_distribution.png
    backtest.png
    hedge_comparison.png
    cvar_weights.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tailrisk.backtest import rolling_var_backtest
from tailrisk.config import EngineConfig
from tailrisk.data import load_prices, log_returns
from tailrisk.hedging import InverseETF, PutOption, evaluate_hedges
from tailrisk.optimization import optimize_cvar
from tailrisk.simulation import MonteCarloEngine, fit_models

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
    }
)

NAVY = "#1f3b73"
TEAL = "#2a9d8f"
ORANGE = "#e76f51"
GOLD = "#e9c46a"
GREY = "#6c757d"

OUT = Path("docs/images")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = EngineConfig.from_yaml("configs/default.yaml")
    prices = load_prices(cfg.portfolio.tickers, cfg.portfolio.start, cfg.portfolio.end)
    rets = log_returns(prices)
    models = fit_models(rets, cfg)
    sim = MonteCarloEngine(cfg, models).run()

    # ---- 1. Fan chart of cumulative portfolio returns ----
    w = np.array(cfg.portfolio.weights)
    portfolio_step = sim.asset_returns @ w  # (n_paths, horizon)
    cum = np.cumsum(portfolio_step, axis=1)  # cumulative log return per path
    horizon = cum.shape[1]
    days = np.arange(1, horizon + 1)
    qs = np.quantile(cum, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(days, qs[0], qs[-1], color=NAVY, alpha=0.10, label="1–99% range")
    ax.fill_between(days, qs[1], qs[-2], color=NAVY, alpha=0.20, label="5–95%")
    ax.fill_between(days, qs[2], qs[-3], color=NAVY, alpha=0.35, label="25–75%")
    ax.plot(days, qs[3], color=NAVY, lw=2, label="median")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title(f"Simulated portfolio paths · {cfg.simulation.n_paths:,} Monte Carlo trajectories")
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Cumulative log return")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.savefig(OUT / "fan_chart.png")
    plt.close(fig)

    # ---- 2. Loss distribution with VaR / CVaR ----
    terminal = cum[:, -1]
    losses = -terminal
    var95 = np.quantile(losses, 0.95)
    var99 = np.quantile(losses, 0.99)
    cvar95 = losses[losses >= var95].mean()
    cvar99 = losses[losses >= var99].mean()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Clip to 1st–99.5th percentile of losses for readability; the tail beyond
    # is exactly what CVaR captures.
    lo, hi = np.quantile(losses, [0.005, 0.995]) * 100
    ax.hist(
        np.clip(losses * 100, lo, hi),
        bins=80, color=NAVY, alpha=0.85, edgecolor="white", linewidth=0.4,
    )
    ax.axvline(var95 * 100, color=GOLD, lw=2, label=f"VaR 95% = {var95:.1%}")
    ax.axvline(cvar95 * 100, color=ORANGE, lw=2, label=f"CVaR 95% = {cvar95:.1%}")
    ax.axvline(cvar99 * 100, color="#b22222", lw=2, label=f"CVaR 99% = {cvar99:.1%}")
    ax.set_title(f"{cfg.simulation.horizon_days}-day loss distribution · fat right tail (1st–99.5pct shown)")
    ax.set_xlabel("Loss (%)  →  bigger losses to the right")
    ax.set_ylabel("Paths")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.savefig(OUT / "loss_distribution.png")
    plt.close(fig)

    # ---- 3. CVaR-optimal weights ----
    asset_terminal = sim.asset_returns.sum(axis=1)
    res = optimize_cvar(
        asset_terminal,
        alpha=cfg.optimization.cvar_alpha,
        long_only=cfg.optimization.long_only,
        max_weight=cfg.optimization.max_weight,
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(sim.tickers))
    w_input = np.array(cfg.portfolio.weights)
    width = 0.4
    ax.bar(x - width / 2, w_input * 100, width, label="Input weights", color=GREY)
    ax.bar(x + width / 2, res.weights * 100, width, label="CVaR-optimal", color=TEAL)
    ax.set_xticks(x)
    ax.set_xticklabels(sim.tickers)
    ax.set_ylabel("Weight (%)")
    ax.set_title(
        f"CVaR-optimal allocation (α={cfg.optimization.cvar_alpha:.0%})  "
        f"·  CVaR={res.cvar:.1%}  E[R]={res.expected_return:.1%}"
    )
    ax.legend()
    fig.savefig(OUT / "cvar_weights.png")
    plt.close(fig)

    # ---- 4. Hedge comparison ----
    ticker_idx = {t: i for i, t in enumerate(sim.tickers)}
    instruments: list = []
    for spec in cfg.hedging.instruments:
        if spec.kind == "put" and spec.underlying in ticker_idx:
            instruments.append(
                PutOption(
                    asset_index=ticker_idx[spec.underlying],
                    moneyness=spec.moneyness or 0.95,
                    premium_pct=spec.premium_pct or 0.012,
                )
            )
        elif spec.kind == "inverse_etf" and spec.ticker in ticker_idx:
            instruments.append(
                InverseETF(
                    asset_index=ticker_idx[spec.ticker],
                    allocation=spec.allocation or 0.05,
                )
            )
    rows = evaluate_hedges(sim.asset_returns, sim.weights, instruments)
    names = [r.name for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    width = 0.27
    ax.bar(x - width, [r.expected_return * 100 for r in rows], width, label="E[R]", color=TEAL)
    ax.bar(x, [r.cvar_95 * 100 for r in rows], width, label="CVaR 95%", color=GOLD)
    ax.bar(x + width, [r.cvar_99 * 100 for r in rows], width, label="CVaR 99%", color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("% over horizon")
    ax.set_title("Hedge strategies · cost vs. tail protection")
    ax.legend()
    fig.savefig(OUT / "hedge_comparison.png")
    plt.close(fig)

    # ---- 5. Backtest plot ----
    weights = np.array(cfg.portfolio.weights)
    portfolio = (rets * weights).sum(axis=1)
    bt = rolling_var_backtest(portfolio, min_train=750, refit_every=63)
    df = bt.to_frame()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df.index, df["realized"] * 100, color=NAVY, lw=0.6, label="Realized return")
    ax.plot(df.index, -df["var_95"] * 100, color=GOLD, lw=1.0, label="−VaR 95%")
    ax.plot(df.index, -df["var_99"] * 100, color=ORANGE, lw=1.0, label="−VaR 99%")
    breaches99 = df[df["breach_99"] == 1]
    ax.scatter(
        breaches99.index, breaches99["realized"] * 100,
        color="red", s=14, marker="x", label=f"99% breach (n={len(breaches99)})",
    )
    cov = bt.coverage_95
    ax.set_title(
        f"Walk-forward VaR backtest · {bt.coverage_95.n:,} OOS days · "
        f"95%: {cov.breaches} breaches ({cov.breach_rate:.2%}), "
        f"Christoffersen p={cov.christoffersen_pvalue:.2f}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily return (%)")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.savefig(OUT / "backtest.png")
    plt.close(fig)

    print(f"Wrote {len(list(OUT.glob('*.png')))} figures to {OUT.resolve()}")


if __name__ == "__main__":
    main()
