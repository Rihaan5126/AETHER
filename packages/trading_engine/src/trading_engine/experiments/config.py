"""Experiment configuration (the single object an experiment is reproduced from)."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading_engine.evolution import EvolutionConfig
from trading_engine.tournaments import EvaluationConfig
from trading_engine.training import SplitConfig

# Relative cost of one fold fit per family (measured on the synthetic dataset;
# only used for a coarse *relative* workload indication, never as a time estimate).
_RELATIVE_FIT_COST = {"momentum": 0.05, "random_forest": 1.0, "xgboost": 0.4, "mlp": 0.6,
                      "random": 0.02, "buy_and_hold": 0.02}


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1, max_length=120)
    dataset_hash: str = Field(..., min_length=64, max_length=64)
    symbol: str = Field(..., min_length=1, max_length=16)
    start_date: date | None = None
    end_date: date | None = None
    seed: int = Field(42, ge=0, le=2**31 - 1)
    split: SplitConfig = SplitConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    evolution: EvolutionConfig = EvolutionConfig()
    max_workers: int = Field(1, ge=1, le=8)

    @model_validator(mode="after")
    def _dates(self) -> ExperimentConfig:
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def estimate_workload(config: ExperimentConfig) -> dict[str, Any]:
    """Counts of model fits and simulations, plus a relative cost index.

    This is deliberately not a runtime estimate: hardware, data length and
    early stopping make wall-clock predictions unreliable.
    """
    evo = config.evolution
    folds = config.evaluation.walk_forward.n_folds
    per_gen = evo.population_size
    families = list(evo.families)
    mean_cost = sum(_RELATIVE_FIT_COST[f] for f in families) / len(families)
    fits = per_gen * evo.generations * folds
    return {
        "agent_evaluations": per_gen * evo.generations,
        "model_fits_upper_bound": fits,
        "simulations_upper_bound": fits + 2 * evo.generations * folds,
        "relative_cost_index": round(fits * mean_cost, 1),
        "note": "elites are cached, so actual fits are usually lower; no runtime estimate",
    }
