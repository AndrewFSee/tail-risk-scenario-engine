"""Stress test scenarios.

Two scenario families:

- *Historical*: replay or block-bootstrap a historical window (e.g. GFC 2008,
  COVID-2020). The output is a sample of return paths drawn from that window.
- *Parametric*: apply named shocks (vol multiplier, mean shift, correlation
  floor) to the fitted models before simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml


@dataclass
class HistoricalScenario:
    name: str
    description: str
    start: str
    end: str
    block_size: int = 5

    def sample_paths(
        self,
        returns: pd.DataFrame,
        n_paths: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Block-bootstrap returns from the historical window.

        Returns a tensor of shape (n_paths, horizon, n_assets) of log returns.
        """
        window = returns.loc[self.start : self.end].values
        T, N = window.shape
        if T < self.block_size:
            raise ValueError(f"Historical window {self.name!r} too short ({T} rows).")

        out = np.empty((n_paths, horizon, N))
        for p in range(n_paths):
            buf = []
            while len(buf) < horizon:
                start = int(rng.integers(0, T - self.block_size + 1))
                buf.extend(window[start : start + self.block_size].tolist())
            out[p] = np.asarray(buf[:horizon])
        return out


@dataclass
class ParametricScenario:
    name: str
    description: str
    vol_multiplier: float = 1.0
    mean_shift_bps: float = 0.0
    correlation_floor: float | None = None


Scenario = HistoricalScenario | ParametricScenario


def load_scenario(path: str | Path) -> Scenario:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    kind: Literal["historical", "parametric"] = data["type"]
    if kind == "historical":
        return HistoricalScenario(
            name=data["name"],
            description=data.get("description", ""),
            start=data["window"]["start"],
            end=data["window"]["end"],
            block_size=int(data.get("block_size", 5)),
        )
    if kind == "parametric":
        s = data.get("shocks", {})
        return ParametricScenario(
            name=data["name"],
            description=data.get("description", ""),
            vol_multiplier=float(s.get("vol_multiplier", 1.0)),
            mean_shift_bps=float(s.get("mean_shift_bps", 0.0)),
            correlation_floor=s.get("correlation_floor"),
        )
    raise ValueError(f"Unknown scenario type: {kind!r}")
