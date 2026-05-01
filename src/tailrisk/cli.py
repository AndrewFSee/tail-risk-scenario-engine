"""Command-line interface for the tail risk engine."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import EngineConfig

app = typer.Typer(add_completion=False, help="Tail risk scenario engine.")
console = Console()


@app.command()
def simulate(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
) -> None:
    """Run the full pipeline: load data, fit models, simulate, report risk."""
    from .data import load_prices, log_returns
    from .risk import tail_metrics
    from .simulation import MonteCarloEngine, fit_models

    cfg = EngineConfig.from_yaml(config)
    console.log(f"Loaded config from {config}")
    prices = load_prices(cfg.portfolio.tickers, cfg.portfolio.start, cfg.portfolio.end)
    rets = log_returns(prices)
    console.log(f"Loaded {len(rets)} return observations across {rets.shape[1]} assets.")

    models = fit_models(rets, cfg)
    engine = MonteCarloEngine(cfg, models)
    result = engine.run()

    table = Table(title=f"Tail metrics  (horizon={cfg.simulation.horizon_days}d, "
                        f"paths={cfg.simulation.n_paths})")
    table.add_column("alpha")
    table.add_column("VaR")
    table.add_column("CVaR")
    table.add_column("mean")
    table.add_column("std")
    table.add_column("excess kurt.")
    for a in cfg.risk.confidence_levels:
        m = tail_metrics(result.portfolio_pnl, alpha=a)
        table.add_row(
            f"{a:.0%}", f"{m.var:.3%}", f"{m.cvar:.3%}",
            f"{m.mean:.3%}", f"{m.std:.3%}", f"{m.excess_kurtosis:.2f}",
        )
    console.print(table)


@app.command()
def optimize(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
) -> None:
    """Solve the CVaR-optimal portfolio over simulated scenarios."""
    from .data import load_prices, log_returns
    from .optimization import optimize_cvar
    from .simulation import MonteCarloEngine, fit_models

    cfg = EngineConfig.from_yaml(config)
    prices = load_prices(cfg.portfolio.tickers, cfg.portfolio.start, cfg.portfolio.end)
    rets = log_returns(prices)
    models = fit_models(rets, cfg)
    sim = MonteCarloEngine(cfg, models).run()

    terminal = sim.asset_returns.sum(axis=1)  # (n_paths, n_assets)
    res = optimize_cvar(
        terminal,
        alpha=cfg.optimization.cvar_alpha,
        long_only=cfg.optimization.long_only,
        max_weight=cfg.optimization.max_weight,
    )

    table = Table(title=f"CVaR-optimal weights  (alpha={cfg.optimization.cvar_alpha:.0%})")
    table.add_column("ticker")
    table.add_column("weight")
    for t, w in zip(sim.tickers, res.weights):
        table.add_row(t, f"{w:.2%}")
    console.print(table)
    console.print(
        f"[bold]CVaR[/bold]={res.cvar:.3%}  "
        f"[bold]VaR[/bold]={res.var:.3%}  "
        f"[bold]E[R][/bold]={res.expected_return:.3%}"
    )


@app.command()
def hedge(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
) -> None:
    """Compare hedge variants vs. the unhedged portfolio."""
    from .data import load_prices, log_returns
    from .hedging import InverseETF, PutOption, evaluate_hedges
    from .simulation import MonteCarloEngine, fit_models

    cfg = EngineConfig.from_yaml(config)
    prices = load_prices(cfg.portfolio.tickers, cfg.portfolio.start, cfg.portfolio.end)
    rets = log_returns(prices)
    models = fit_models(rets, cfg)
    sim = MonteCarloEngine(cfg, models).run()

    instruments: list = []
    ticker_idx = {t: i for i, t in enumerate(sim.tickers)}
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
    table = Table(title="Hedge comparison")
    table.add_column("strategy")
    table.add_column("E[R]")
    table.add_column("VaR 95%")
    table.add_column("CVaR 95%")
    table.add_column("CVaR 99%")
    for r in rows:
        table.add_row(
            r.name, f"{r.expected_return:.3%}", f"{r.var_95:.3%}",
            f"{r.cvar_95:.3%}", f"{r.cvar_99:.3%}",
        )
    console.print(table)


@app.command()
def backtest(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    min_train: int = typer.Option(500, help="Minimum training observations."),
    refit_every: int = typer.Option(63, help="Refit GARCH every K days."),
) -> None:
    """Walk-forward 1-day VaR backtest with Kupiec & Christoffersen tests."""
    import numpy as np

    from .backtest import rolling_var_backtest
    from .data import load_prices, log_returns

    cfg = EngineConfig.from_yaml(config)
    prices = load_prices(cfg.portfolio.tickers, cfg.portfolio.start, cfg.portfolio.end)
    rets = log_returns(prices)
    weights = np.array(cfg.portfolio.weights)
    portfolio = (rets * weights).sum(axis=1)
    console.log(f"Backtesting {len(portfolio)} obs (out-of-sample after {min_train}).")

    result = rolling_var_backtest(
        portfolio, min_train=min_train, refit_every=refit_every
    )

    for label, cov in [("95%", result.coverage_95), ("99%", result.coverage_99)]:
        verdict = "[green]PASS[/green]" if cov.passes() else "[red]FAIL[/red]"
        table = Table(title=f"VaR {label} backtest  {verdict}")
        table.add_column("metric")
        table.add_column("value")
        table.add_row("forecasts (n)", f"{cov.n}")
        table.add_row("breaches", f"{cov.breaches}")
        table.add_row("expected breaches", f"{cov.expected_breaches:.1f}")
        table.add_row("breach rate", f"{cov.breach_rate:.3%}")
        table.add_row("nominal rate", f"{(1 - cov.alpha):.3%}")
        table.add_row("Kupiec LR (p)", f"{cov.kupiec_lr:.2f}  ({cov.kupiec_pvalue:.3f})")
        table.add_row(
            "Christoffersen LR (p)",
            f"{cov.christoffersen_lr:.2f}  ({cov.christoffersen_pvalue:.3f})",
        )
        table.add_row("Conditional cov. LR (p)", f"{cov.cc_lr:.2f}  ({cov.cc_pvalue:.3f})")
        console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()

