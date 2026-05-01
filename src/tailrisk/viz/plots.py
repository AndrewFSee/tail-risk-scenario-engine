"""Plotly figures for the dashboard."""

from __future__ import annotations

import numpy as np


def loss_distribution_figure(pnl: np.ndarray, var: float, cvar: float):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pnl, nbinsx=80, name="P&L"))
    fig.add_vline(x=-var, line_dash="dash", annotation_text=f"VaR={var:.2%}")
    fig.add_vline(x=-cvar, line_dash="dot", annotation_text=f"CVaR={cvar:.2%}")
    fig.update_layout(
        title="Portfolio P&L Distribution",
        xaxis_title="Return",
        yaxis_title="Count",
        bargap=0.02,
    )
    return fig


def fan_chart_figure(asset_paths: np.ndarray, weights: np.ndarray, tickers: list[str]):
    import plotly.graph_objects as go

    portfolio_paths = asset_paths @ weights  # (n_paths, horizon)
    q = np.quantile(portfolio_paths, [0.05, 0.5, 0.95], axis=0)
    t = np.arange(portfolio_paths.shape[1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=q[2], name="95th", line=dict(width=0)))
    fig.add_trace(
        go.Scatter(x=t, y=q[0], name="5th", fill="tonexty", line=dict(width=0))
    )
    fig.add_trace(go.Scatter(x=t, y=q[1], name="Median"))
    fig.update_layout(
        title="Cumulative Portfolio Return Fan",
        xaxis_title="Day",
        yaxis_title="Cumulative log return",
    )
    return fig


def hedge_comparison_figure(rows: list):
    import plotly.graph_objects as go

    names = [r.name for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[r.cvar_95 for r in rows], name="CVaR 95%"))
    fig.add_trace(go.Bar(x=names, y=[r.cvar_99 for r in rows], name="CVaR 99%"))
    fig.add_trace(
        go.Bar(x=names, y=[r.expected_return for r in rows], name="Expected return")
    )
    fig.update_layout(barmode="group", title="Hedge Strategy Comparison")
    return fig


def backtest_figure(result):
    """Plot realized returns vs. predicted VaR band with breach markers."""
    import plotly.graph_objects as go

    df = result.to_frame()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["realized"], mode="lines",
            name="Realized return", line=dict(width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=-df["var_95"], mode="lines", name="-VaR 95%",
            line=dict(dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=-df["var_99"], mode="lines", name="-VaR 99%",
            line=dict(dash="dot"),
        )
    )
    breaches = df[df["breach_99"] == 1]
    if len(breaches):
        fig.add_trace(
            go.Scatter(
                x=breaches.index, y=breaches["realized"],
                mode="markers", name="99% breach",
                marker=dict(color="red", size=8, symbol="x"),
            )
        )
    fig.update_layout(
        title="VaR Backtest — realized returns vs. forecasts",
        xaxis_title="Date",
        yaxis_title="Daily log return",
    )
    return fig
