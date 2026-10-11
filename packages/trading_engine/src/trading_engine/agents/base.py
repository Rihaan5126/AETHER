"""Agent abstraction.

The decision pipeline is split into explicit layers so that a model prediction is
never mistaken for an executable trade::

    Observations (point-in-time features)
        -> forecast()        model output, e.g. P(up) -- *not* a calibrated probability
        -> position policy   forecast -> target weight in [0, max_allocation]
        -> AgentDecision     (symbol, target_weight, timestamp)
        -> RiskManager       (simulator) may clip/reject
        -> execution         (simulator) next open, with costs

Agents never see the simulator, cash or the risk limits, and cannot bypass them.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel

from trading_engine.agents.genomes import BENCHMARK_FAMILIES, genome_to_dict, parse_genome


@dataclass(frozen=True)
class AgentDecision:
    symbol: str
    target_weight: float
    timestamp: datetime
    confidence: float | None = None  # raw model score; not a calibrated probability


@dataclass(frozen=True)
class Observations:
    """What an agent may look at when deciding for the sessions in ``features.index``.

    ``history`` holds raw bars up to and including the last decision date (it may
    start earlier, so rule-based agents can compute long windows). Nothing after
    ``features.index[-1]`` is ever included.
    """

    symbol: str
    features: pd.DataFrame
    history: pd.DataFrame

    def __post_init__(self) -> None:
        if (
            len(self.features)
            and len(self.history)
            and (self.history.index[-1] > self.features.index[-1])
        ):
            raise ValueError("observation history extends beyond the last decision date")


@dataclass(frozen=True)
class TrainingSet:
    X: pd.DataFrame
    y: pd.Series
    history: pd.DataFrame


@dataclass(frozen=True)
class FitContext:
    seed: int
    horizon: int = 1
    max_seconds: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseTradingAgent(ABC):
    family: ClassVar[str]
    is_learned: ClassVar[bool] = False

    def __init__(self, agent_id: str, genome: BaseModel, seed: int) -> None:
        if getattr(genome, "family", None) != self.family:
            raise TypeError(f"{type(self).__name__} requires a {self.family} genome")
        self.agent_id = agent_id
        self.genome = genome
        self.seed = int(seed)
        self.fitted = not self.is_learned
        self.training_metrics: dict[str, Any] = {}

    @property
    def is_benchmark(self) -> bool:
        return self.family in BENCHMARK_FAMILIES

    # ---- learning ---------------------------------------------------------------
    def fit(self, training: TrainingSet, context: FitContext) -> dict[str, Any]:
        """Fit on training rows only. Baselines record a no-op fit."""
        self.training_metrics = {"n_samples": len(training.X), "fitted": False}
        self.fitted = True
        return self.training_metrics

    # ---- decision layers ----------------------------------------------------------
    @abstractmethod
    def forecast(self, observations: Observations) -> pd.Series:
        """Raw model output per decision date (NaN where undefined)."""

    def position_policy(self, forecast: pd.Series) -> pd.Series:
        """Map the forecast to a target weight. Default: forecast is already a weight."""
        return forecast.clip(0.0, 1.0)

    def predict(self, observations: Observations) -> np.ndarray:
        """Target weights aligned with ``observations.features.index``."""
        if not self.fitted:
            raise RuntimeError(f"agent {self.agent_id} must be fitted before predicting")
        fc = self.forecast(observations).reindex(observations.features.index)
        return self.position_policy(fc).to_numpy(dtype="float64")

    def decisions(self, observations: Observations) -> list[AgentDecision]:
        weights = self.predict(observations)
        fc = self.forecast(observations).reindex(observations.features.index)
        return [
            AgentDecision(
                observations.symbol,
                float(w),
                ts.to_pydatetime(),
                float(c) if self.is_learned and np.isfinite(c) else None,
            )
            for ts, w, c in zip(observations.features.index, weights, fc.to_numpy(), strict=True)
        ]

    # ---- metadata & persistence ------------------------------------------------------
    def get_metadata(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "family": self.family,
            "is_learned": self.is_learned,
            "is_benchmark": self.is_benchmark,
            "seed": self.seed,
            "genome": genome_to_dict(self.genome),
            "training_metrics": self.training_metrics,
        }

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "meta.json").write_text(json.dumps(self.get_metadata(), indent=2, default=str))
        self._save_model(directory)

    def _save_model(self, directory: Path) -> None:
        """Learned agents write their model files here."""

    def load(self, directory: Path) -> None:
        meta = json.loads((directory / "meta.json").read_text())
        if meta["family"] != self.family:
            raise ValueError("artifact family mismatch")
        if parse_genome(meta["genome"]) != self.genome:
            raise ValueError("artifact genome does not match this agent")
        self.training_metrics = meta.get("training_metrics", {})
        self._load_model(directory)
        self.fitted = True

    def _load_model(self, directory: Path) -> None:
        """Learned agents restore their model files here."""
