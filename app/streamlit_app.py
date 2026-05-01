"""Streamlit dashboard for the tail risk engine.

Run with:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from tailrisk.config import EngineConfig
from tailrisk.data import load_prices, log_returns
from tailrisk.hedging import InverseETF, PutOption, evaluate_hedges
from tailrisk.optimization import optimize_cvar
from tailrisk.risk import tail_metrics
from tailrisk.simulation import MonteCarloEngine, fit_models
from tailrisk.viz import (
    backtest_figure,
    fan_chart_figure,
    hedge_comparison_figure,
    loss_distribution_figure,
)

st.set_page_config(page_title="Tail Risk Engine", layout="wide")
st.title("Tail Risk Scenario Engine")

config_path = st.sidebar.text_input("Config path", "configs/default.yaml")
if not Path(config_path).exists():
    st.warning(f"Config not found: {config_path}")
    st.stop()

cfg = EngineConfig.from_yaml(config_path)

st.sidebar.subheader("Simulation")
n_paths = st.sidebar.number_input("Paths", 1_000, 200_000, value=cfg.simulation.n_paths, step=1_000)
horizon = st.sidebar.number_input("Horizon (days)", 1, 252, value=cfg.simulation.horizon_days)
cfg.simulation.n_paths = int(n_paths)
cfg.simulation.horizon_days = int(horizon)

with st.spinner("Loading prices..."):
    prices = load_prices(cfg.portfolio.tickers, cfg.portfolio.start, cfg.portfolio.end)
    rets = log_returns(prices)

with st.spinner("Fitting models..."):
    models = fit_models(rets, cfg)

with st.spinner("Simulating..."):
    sim = MonteCarloEngine(cfg, models).run()

col1, col2 = st.columns(2)
with col1:
    m = tail_metrics(sim.portfolio_pnl, alpha=0.95)
    st.metric("VaR 95%", f"{m.var:.2%}")
    st.metric("CVaR 95%", f"{m.cvar:.2%}")
with col2:
    m99 = tail_metrics(sim.portfolio_pnl, alpha=0.99)
    st.metric("VaR 99%", f"{m99.var:.2%}")
    st.metric("CVaR 99%", f"{m99.cvar:.2%}")

st.plotly_chart(
    loss_distribution_figure(sim.portfolio_pnl, m.var, m.cvar),
    use_container_width=True,
)
st.plotly_chart(
    fan_chart_figure(sim.asset_paths, sim.weights, sim.tickers),
    use_container_width=True,
)

st.subheader("CVaR-optimal portfolio")
if st.button("Optimize"):
    terminal = sim.asset_returns.sum(axis=1)
    res = optimize_cvar(
        terminal,
        alpha=cfg.optimization.cvar_alpha,
        long_only=cfg.optimization.long_only,
        max_weight=cfg.optimization.max_weight,
    )
    st.write({t: float(w) for t, w in zip(sim.tickers, res.weights)})
    st.write(f"CVaR: {res.cvar:.3%} | VaR: {res.var:.3%} | E[R]: {res.expected_return:.3%}")

st.subheader("Hedge comparison")
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
st.plotly_chart(hedge_comparison_figure(rows), use_container_width=True)

st.subheader("Ask a question")
q = st.text_input(
    "e.g. 'What is my loss if 2008 repeats?', "
    "'What if tech crashes 20%?', or 'What if volatility doubles?'"
)
if q:
    from tailrisk.llm import answer

    try:
        result = answer(q, sim, historical_returns=rets)
    except ValueError as e:
        st.error(str(e))
    else:
        st.markdown(f"**{result.summary}**")
        if result.metrics:
            cols = st.columns(4)
            cols[0].metric("E[R]", f"{result.metrics['mean']:.2%}")
            cols[1].metric("VaR 95%", f"{result.metrics['var_95']:.2%}")
            cols[2].metric("CVaR 95%", f"{result.metrics['cvar_95']:.2%}")
            cols[3].metric("CVaR 99%", f"{result.metrics['cvar_99']:.2%}")
        if result.pnl is not None:
            st.plotly_chart(
                loss_distribution_figure(
                    result.pnl,
                    result.metrics["var_95"],
                    result.metrics["cvar_95"],
                ),
                use_container_width=True,
            )
        with st.expander("Parsed request"):
            st.write(result.request)

st.subheader("VaR backtest")
st.caption(
    "Walk-forward 1-day VaR on the historical portfolio with Kupiec & "
    "Christoffersen coverage tests. Refits a portfolio-level GARCH(1,1)-t "
    "every K trading days on an expanding window."
)
col_a, col_b, _ = st.columns([1, 1, 3])
min_train = col_a.number_input("Min train (days)", min_value=250, max_value=2000, value=750, step=50)
refit_every = col_b.number_input("Refit every (days)", min_value=21, max_value=252, value=63, step=21)

if st.button("Run backtest"):
    from tailrisk.backtest import rolling_var_backtest

    weights = np.array(cfg.portfolio.weights)
    portfolio = (rets * weights).sum(axis=1)
    with st.spinner(f"Running walk-forward backtest on {len(portfolio)} obs..."):
        bt = rolling_var_backtest(
            portfolio, min_train=int(min_train), refit_every=int(refit_every)
        )
    for label, cov in [("95%", bt.coverage_95), ("99%", bt.coverage_99)]:
        st.markdown(f"**VaR {label}** — "
                    f"breaches {cov.breaches} / expected {cov.expected_breaches:.1f} "
                    f"({cov.breach_rate:.2%} vs nominal {(1-cov.alpha):.0%})")
        cols = st.columns(3)
        cols[0].metric("Kupiec p-value", f"{cov.kupiec_pvalue:.3f}")
        cols[1].metric("Christoffersen p-value", f"{cov.christoffersen_pvalue:.3f}")
        cols[2].metric("Conditional cov. p-value", f"{cov.cc_pvalue:.3f}")
    st.plotly_chart(backtest_figure(bt), use_container_width=True)
