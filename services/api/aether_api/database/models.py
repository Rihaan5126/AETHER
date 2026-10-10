"""Relational schema (PostgreSQL in deployment; SQLite for tests).

Large time series (equity curves) are stored as Parquet artifacts referenced
from ``equity_curves``; model weights are files referenced from ``agents``.
Completed experiment records are never deleted: failed and cancelled runs stay
visible.
"""

from __future__ import annotations

import secrets
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JSONType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(6)}"


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSONType, list[Any]: JSONType}


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    source: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(16))
    hash: Mapped[str] = mapped_column(String(64), unique=True)
    symbols: Mapped[list[Any]] = mapped_column(JSONType)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    rows: Mapped[int] = mapped_column(Integer)
    params: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    validation: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Experiment(Base):
    __tablename__ = "experiments"
    __table_args__ = (Index("ix_experiments_status_created", "status", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    kind: Mapped[str] = mapped_column(String(16), default="evolution")
    config: Mapped[dict[str, Any]] = mapped_column(JSONType)
    manifest: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), index=True)
    status: Mapped[str] = mapped_column(String(16), default="draft")
    seed: Mapped[int] = mapped_column(Integer)
    code_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True, nullable=True)
    current_generation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    development_period: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    test_period: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    test_contaminated: Mapped[bool] = mapped_column(Boolean, default=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    wall_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    resource_usage: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    dataset: Mapped[Dataset] = relationship()


class Agent(Base):
    __tablename__ = "agents"
    __table_args__ = (Index("ix_agents_experiment_generation", "experiment_id", "generation"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    algorithm: Mapped[str] = mapped_column(String(32), index=True)
    genome: Mapped[dict[str, Any]] = mapped_column(JSONType)
    generation: Mapped[int] = mapped_column(Integer)
    parent_id: Mapped[str | None] = mapped_column(
        ForeignKey("agents.id"), nullable=True, index=True
    )
    origin: Mapped[str] = mapped_column(String(16))
    seed: Mapped[int] = mapped_column(Integer)
    is_benchmark: Mapped[bool] = mapped_column(Boolean, default=False)
    artifact_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    artifact_files: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"), index=True)
    tournament_id: Mapped[str | None] = mapped_column(ForeignKey("tournaments.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(16))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    training_metrics: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class Tournament(Base):
    __tablename__ = "tournaments"
    __table_args__ = (UniqueConstraint("experiment_id", "generation"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    generation: Mapped[int] = mapped_column(Integer)
    period: Mapped[str] = mapped_column(String(16), default="validation")
    evaluation_config: Mapped[dict[str, Any]] = mapped_column(JSONType)
    config_fingerprint: Mapped[str] = mapped_column(String(32))
    folds: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    benchmark_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="running")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    results: Mapped[list[TournamentResult]] = relationship(
        back_populates="tournament", order_by="TournamentResult.rank"
    )


class TournamentResult(Base):
    __tablename__ = "tournament_results"
    __table_args__ = (UniqueConstraint("tournament_id", "agent_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    tournament_id: Mapped[str] = mapped_column(ForeignKey("tournaments.id"), index=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"), index=True)
    total_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32))
    status_reason: Mapped[str] = mapped_column(Text, default="")
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    fold_metrics: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    score_components: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    warnings: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    risk_events: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    evaluation_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    tournament: Mapped[Tournament] = relationship(back_populates="results")
    agent: Mapped[Agent] = relationship()


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tournament_result_id: Mapped[str] = mapped_column(
        ForeignKey("tournament_results.id"), index=True
    )
    symbol: Mapped[str] = mapped_column(String(16))
    timestamp: Mapped[datetime] = mapped_column(DateTime)  # execution time (exchange local)
    signal_time: Mapped[datetime] = mapped_column(DateTime)
    side: Mapped[str] = mapped_column(String(4))
    quantity: Mapped[float] = mapped_column(Float)
    execution_price: Mapped[float] = mapped_column(Float)
    reference_price: Mapped[float] = mapped_column(Float)
    fees: Mapped[float] = mapped_column(Float)
    slippage: Mapped[float] = mapped_column(Float)
    notional: Mapped[float] = mapped_column(Float)
    target_weight: Mapped[float] = mapped_column(Float)


class EquityCurve(Base):
    __tablename__ = "equity_curves"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    tournament_id: Mapped[str | None] = mapped_column(
        ForeignKey("tournaments.id"), index=True, nullable=True
    )
    final_test_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("final_test_runs.id"), nullable=True
    )
    path: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64))
    rows: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelLineage(Base):
    __tablename__ = "model_lineage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    child_id: Mapped[str] = mapped_column(ForeignKey("agents.id"), index=True)
    parent_id: Mapped[str | None] = mapped_column(
        ForeignKey("agents.id"), index=True, nullable=True
    )
    generation: Mapped[int] = mapped_column(Integer)
    operation: Mapped[str] = mapped_column(String(16))
    mutation_description: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    genome_before: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    genome_after: Mapped[dict[str, Any]] = mapped_column(JSONType)
    training_seed: Mapped[int] = mapped_column(Integer)
    validation_score: Mapped[float | None] = mapped_column(Float, nullable=True)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_status", "status"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    job_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    experiment_id: Mapped[str | None] = mapped_column(
        ForeignKey("experiments.id"), nullable=True, index=True
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class FinalTestRun(Base):
    __tablename__ = "final_test_runs"
    __table_args__ = (UniqueConstraint("experiment_id", "run_number"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    run_number: Mapped[int] = mapped_column(Integer)
    agent_ids: Mapped[list[Any]] = mapped_column(JSONType)
    status: Mapped[str] = mapped_column(String(16), default="running")
    results: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    benchmark_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    test_period: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    acknowledged_contamination: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class OptimizationStudy(Base):
    __tablename__ = "optimization_studies"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    family: Mapped[str] = mapped_column(String(32))
    config: Mapped[dict[str, Any]] = mapped_column(JSONType)
    status: Mapped[str] = mapped_column(String(16), default="running")
    best_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_params: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    best_genome: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    trials: Mapped[list[OptimizationTrial]] = relationship(order_by="OptimizationTrial.number")


class OptimizationTrial(Base):
    __tablename__ = "optimization_trials"
    __table_args__ = (UniqueConstraint("study_id", "number"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_id: Mapped[str] = mapped_column(ForeignKey("optimization_studies.id"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(16))
    params: Mapped[dict[str, Any]] = mapped_column(JSONType)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    fold_scores: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
