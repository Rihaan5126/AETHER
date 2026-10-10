from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from aether_api.schemas.common import APIModel, UTCDateTime

Family = Literal["momentum", "random_forest", "xgboost", "mlp"]


class ExperimentCreateRequest(BaseModel):
    """User-facing experiment settings, mapped onto the engine's ExperimentConfig."""

    name: str = Field(..., min_length=1, max_length=120)
    dataset_id: str
    symbol: str = Field(..., min_length=1, max_length=16)
    start_date: date | None = None
    end_date: date | None = None
    families: list[Family] = Field(
        default_factory=lambda: ["momentum", "random_forest", "xgboost", "mlp"], min_length=1)
    population_size: int = Field(8, ge=2, le=64)
    generations: int = Field(3, ge=1, le=20)
    elite_count: int = Field(2, ge=0, le=16)
    immigrant_count: int = Field(1, ge=0, le=8)
    initial_capital: float = Field(10_000, gt=0, le=1e9)
    fee_bps: float = Field(1.0, ge=0, le=500)
    slippage_bps: float = Field(5.0, ge=0, le=500)
    max_position: float = Field(1.0, gt=0, le=1)
    drawdown_circuit_breaker: float | None = Field(None, gt=0, lt=1)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    n_folds: int = Field(3, ge=1, le=8)
    test_fraction: float = Field(0.2, gt=0.05, lt=0.5)
    test_start: date | None = None
    max_fit_seconds: float = Field(120, gt=0, le=3600, description="training budget per fit")

    @model_validator(mode="after")
    def _check(self) -> ExperimentCreateRequest:
        if self.elite_count >= self.population_size:
            raise ValueError("elite_count must be smaller than population_size")
        if self.elite_count + self.immigrant_count > self.population_size:
            raise ValueError("elite_count + immigrant_count exceeds population_size")
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        return self


class GenerationSummary(BaseModel):
    generation: int
    tournament_id: str
    status: str
    n_agents: int
    best_score: float | None
    median_score: float | None
    best_agent_id: str | None
    random_baseline_score: float | None
    buy_and_hold_return: float | None


class ExperimentOut(APIModel):
    id: str
    name: str
    kind: str
    status: str
    dataset_id: str
    symbol: str
    seed: int
    current_generation: int | None
    planned_generations: int
    population_size: int
    progress: float | None
    test_contaminated: bool
    created_at: UTCDateTime
    started_at: UTCDateTime | None
    completed_at: UTCDateTime | None


class ExperimentDetail(ExperimentOut):
    config: dict[str, Any]
    manifest: dict[str, Any] | None
    code_version: str | None
    development_period: dict[str, Any] | None
    test_period: dict[str, Any] | None
    error_message: str | None
    wall_seconds: float | None
    resource_usage: dict[str, Any] | None
    workload: dict[str, Any]
    generations: list[GenerationSummary]
    jobs: list[dict[str, Any]]
    final_test_runs: list[dict[str, Any]]
    optimization_studies: list[dict[str, Any]]


class FinalTestRequest(BaseModel):
    agent_ids: list[str] | None = Field(None, max_length=5)
    acknowledge_contamination: bool = False


class OptimizeRequest(BaseModel):
    family: Family = "xgboost"
    n_trials: int = Field(15, ge=1, le=100)
    timeout_seconds: float = Field(900, gt=0, le=6 * 3600)
