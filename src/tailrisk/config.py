"""Configuration schema for the tail risk engine.

Uses pydantic for validation so YAML configs can be safely loaded and passed
between modules without ad-hoc dict access.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


Marginal = Literal["normal", "student_t", "gpd_evt"]
VolModel = Literal["constant", "garch"]
RegimeModel = Literal["none", "markov_2state"]
Dependence = Literal["gaussian", "student_t_copula"]


class PortfolioConfig(BaseModel):
    tickers: list[str]
    weights: list[float]
    start: date
    end: date | None = None

    @field_validator("weights")
    @classmethod
    def _check_weights(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("weights must be non-empty")
        s = sum(v)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0 (got {s})")
        return v


class SimulationConfig(BaseModel):
    n_paths: int = 10_000
    horizon_days: int = 21
    seed: int = 42
    marginal: Marginal = "student_t"
    volatility: VolModel = "garch"
    regime: RegimeModel = "markov_2state"
    dependence: Dependence = "student_t_copula"
    # Cap on |standardized innovation| to prevent numerical blow-up of
    # per-path GARCH variance under very fat-tailed Student-t draws. Set to
    # None to disable. ~10 still allows >6-sigma events.
    max_innovation_z: float | None = 10.0


class RiskConfig(BaseModel):
    confidence_levels: list[float] = Field(default_factory=lambda: [0.95, 0.99])


class OptimizationConfig(BaseModel):
    enabled: bool = True
    objective: Literal["min_cvar", "min_variance"] = "min_cvar"
    cvar_alpha: float = 0.95
    long_only: bool = True
    max_weight: float = 1.0


class HedgeInstrument(BaseModel):
    kind: Literal["put", "inverse_etf"]
    underlying: str | None = None
    ticker: str | None = None
    moneyness: float | None = None
    tenor_days: int | None = None
    premium_pct: float | None = None
    allocation: float | None = None


class HedgingConfig(BaseModel):
    enabled: bool = False
    instruments: list[HedgeInstrument] = Field(default_factory=list)


class EngineConfig(BaseModel):
    portfolio: PortfolioConfig
    simulation: SimulationConfig = SimulationConfig()
    risk: RiskConfig = RiskConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    hedging: HedgingConfig = HedgingConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EngineConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(**data)
