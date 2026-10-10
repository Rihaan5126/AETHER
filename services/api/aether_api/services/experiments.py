"""Experiment lifecycle: create (idempotent), start, cancel, final test, optimisation."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aether_api.core.config import Settings
from aether_api.core.errors import APIError, bad_request, conflict, not_found
from aether_api.database.models import (
    Agent,
    Dataset,
    Experiment,
    FinalTestRun,
    Job,
    OptimizationStudy,
    Tournament,
    TournamentResult,
    new_id,
)
from aether_api.schemas.experiments import (
    ExperimentCreateRequest,
    FinalTestRequest,
    OptimizeRequest,
)
from aether_api.schemas.tournaments import TournamentCreateRequest
from aether_api.services.jobs import TERMINAL, create_job, request_cancel
from trading_engine.agents import parse_genome
from trading_engine.evolution import EvolutionConfig
from trading_engine.experiments import ExperimentConfig
from trading_engine.optimization import OptimizationConfig
from trading_engine.risk import RiskConfig
from trading_engine.simulation import ExecutionConfig
from trading_engine.tournaments import EvaluationConfig
from trading_engine.training import SplitConfig, WalkForwardConfig

ACTIVE = {"queued", "running"}


def _dataset(db: Session, dataset_id: str, symbol: str) -> Dataset:
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise not_found("dataset", dataset_id)
    if symbol.upper() not in ds.symbols:
        raise bad_request(f"symbol {symbol!r} is not in dataset {dataset_id}",
                          {"available": ds.symbols})
    return ds


def build_config(req: ExperimentCreateRequest, ds: Dataset, settings: Settings) -> ExperimentConfig:
    if req.population_size > settings.max_population:
        raise bad_request(f"population_size is limited to {settings.max_population}")
    if req.generations > settings.max_generations:
        raise bad_request(f"generations is limited to {settings.max_generations}")
    evaluation = EvaluationConfig(
        execution=ExecutionConfig(initial_capital=req.initial_capital, fee_bps=req.fee_bps,
                                  slippage_bps=req.slippage_bps),
        risk=RiskConfig(max_position=req.max_position,
                        drawdown_circuit_breaker=req.drawdown_circuit_breaker),
        walk_forward=WalkForwardConfig(n_folds=req.n_folds),
        max_fit_seconds=req.max_fit_seconds,
    )
    try:
        return ExperimentConfig(
            name=req.name, dataset_hash=ds.hash, symbol=req.symbol.upper(),
            start_date=req.start_date, end_date=req.end_date, seed=req.seed,
            split=SplitConfig(test_fraction=req.test_fraction, test_start=req.test_start),
            evaluation=evaluation,
            evolution=EvolutionConfig(
                population_size=req.population_size, generations=req.generations,
                elite_count=req.elite_count, immigrant_count=req.immigrant_count,
                families=tuple(dict.fromkeys(req.families))),
            max_workers=settings.max_workers_per_job,
        )
    except ValueError as exc:
        raise bad_request(str(exc)) from exc


def _create(db: Session, *, name: str, kind: str, ds: Dataset, config: ExperimentConfig,
            idempotency_key: str | None) -> tuple[Experiment, bool]:
    stored = config.to_json_dict()
    if idempotency_key:
        existing = db.scalar(select(Experiment).where(
            Experiment.idempotency_key == idempotency_key))
        if existing is not None:
            if existing.config != stored or existing.kind != kind:
                raise conflict("Idempotency-Key was already used with a different request")
            return existing, False
    exp = Experiment(id=new_id("exp"), name=name, kind=kind, config=stored, dataset_id=ds.id,
                     status="draft", seed=config.seed, idempotency_key=idempotency_key)
    db.add(exp)
    db.flush()
    return exp, True


def create_experiment(db: Session, req: ExperimentCreateRequest, settings: Settings,
                      idempotency_key: str | None) -> tuple[Experiment, bool]:
    ds = _dataset(db, req.dataset_id, req.symbol)
    config = build_config(req, ds, settings)
    return _create(db, name=req.name, kind="evolution", ds=ds, config=config,
                   idempotency_key=idempotency_key)


def create_tournament_experiment(db: Session, req: TournamentCreateRequest, settings: Settings,
                                 idempotency_key: str | None) -> tuple[Experiment, bool]:
    ds = _dataset(db, req.dataset_id, req.symbol)
    agents = []
    for choice in req.agents:
        try:
            genome = parse_genome({**(choice.genome or {}), "family": choice.family})
        except ValueError as exc:
            raise bad_request(f"invalid genome for {choice.family}: {exc}") from exc
        agents.append({"family": choice.family, "genome": genome.model_dump(mode="json"),
                       "seed": choice.seed})
    base = ExperimentCreateRequest(name=req.name, dataset_id=req.dataset_id, symbol=req.symbol,
                                   initial_capital=req.initial_capital, fee_bps=req.fee_bps,
                                   slippage_bps=req.slippage_bps, seed=req.seed,
                                   n_folds=req.n_folds, test_fraction=req.test_fraction,
                                   generations=1)
    config = build_config(base, ds, settings).model_copy(update={"agents": agents})
    config = ExperimentConfig.model_validate(config.model_dump())
    return _create(db, name=req.name, kind="tournament", ds=ds, config=config,
                   idempotency_key=idempotency_key)


def active_job(db: Session, experiment_id: str, job_type: str | None = None) -> Job | None:
    q = select(Job).where(Job.experiment_id == experiment_id, Job.status.in_(ACTIVE))
    if job_type:
        q = q.where(Job.job_type == job_type)
    return db.scalar(q.order_by(Job.created_at.desc()))


def start_experiment(db: Session, exp: Experiment) -> tuple[Job, bool]:
    """Returns (job, created). Starting an already running experiment is idempotent."""
    running = active_job(db, exp.id, "run_experiment")
    if running is not None:
        return running, False
    if exp.status not in {"draft"}:
        raise conflict(f"experiment is {exp.status}; create a new experiment to re-run it "
                       "(completed records are immutable)")
    job = create_job(db, "run_experiment", {"experiment_id": exp.id}, experiment_id=exp.id)
    exp.status = "queued"
    return job, True


def cancel_experiment(db: Session, exp: Experiment) -> Job | None:
    job = active_job(db, exp.id)
    if job is None:
        if exp.status in {"draft", "queued"}:
            exp.status = "cancelled"
            return None
        raise conflict(f"experiment is {exp.status}; nothing to cancel")
    request_cancel(db, job)
    return job


def default_final_test_agents(db: Session, exp: Experiment) -> list[str]:
    last = db.scalar(select(Tournament).where(Tournament.experiment_id == exp.id)
                     .order_by(Tournament.generation.desc()))
    if last is None:
        return []
    rows = db.scalars(select(TournamentResult).where(TournamentResult.tournament_id == last.id)
                      .order_by(TournamentResult.rank)).all()
    picks = [r.agent_id for r in rows if r.status == "eligible"][:3]
    picks += [r.agent_id for r in rows if r.agent.algorithm == "random"][:1]
    return picks


def request_final_test(db: Session, exp: Experiment, req: FinalTestRequest) -> tuple[FinalTestRun,
                                                                                    Job]:
    if exp.status != "completed":
        raise conflict("the final test can only run after the experiment has completed")
    previous = db.scalar(select(func.count()).select_from(FinalTestRun)
                         .where(FinalTestRun.experiment_id == exp.id))
    if previous and not req.acknowledge_contamination:
        raise APIError(409, "final_test_locked",
                       "the final test has already been run for this experiment. Running it "
                       "again makes the test period part of development history; resend with "
                       "acknowledge_contamination=true to proceed.")
    agent_ids = req.agent_ids or default_final_test_agents(db, exp)
    if not agent_ids:
        raise bad_request("no agents available for the final test")
    final_gen = exp.current_generation
    for aid in agent_ids:
        agent = db.get(Agent, aid)
        if agent is None or agent.experiment_id != exp.id:
            raise bad_request(f"agent {aid} does not belong to this experiment")
        if agent.generation != final_gen or not agent.artifact_path:
            raise bad_request(f"agent {aid} is not in the final generation (only final-"
                              "generation agents have stored development-fitted models)")
    run = FinalTestRun(id=new_id("ftr"), experiment_id=exp.id, run_number=(previous or 0) + 1,
                       agent_ids=agent_ids, status="queued",
                       acknowledged_contamination=bool(previous))
    db.add(run)
    if previous:
        exp.test_contaminated = True
    db.flush()
    job = create_job(db, "final_test", {"final_test_run_id": run.id}, experiment_id=exp.id)
    return run, job


def request_optimization(db: Session, exp: Experiment, req: OptimizeRequest,
                         settings: Settings) -> tuple[OptimizationStudy, Job]:
    if req.n_trials > settings.max_optuna_trials:
        raise bad_request(f"n_trials is limited to {settings.max_optuna_trials}")
    if exp.status != "completed":
        raise conflict("run optimisation after the experiment has completed")
    if active_job(db, exp.id, "optimize") is not None:
        raise conflict("an optimisation study is already running for this experiment")
    cfg = OptimizationConfig(family=req.family, n_trials=req.n_trials,
                             timeout_seconds=req.timeout_seconds, seed=exp.seed)
    study = OptimizationStudy(id=new_id("opt"), experiment_id=exp.id, family=req.family,
                              config=cfg.model_dump(mode="json"), status="queued")
    db.add(study)
    db.flush()
    job = create_job(db, "optimize", {"study_id": study.id}, experiment_id=exp.id)
    return study, job


def latest_job_progress(db: Session, exp: Experiment) -> float | None:
    job = db.scalar(select(Job).where(Job.experiment_id == exp.id,
                                      Job.job_type == "run_experiment")
                    .order_by(Job.created_at.desc()))
    if job is None:
        return None
    return 1.0 if job.status == "completed" else job.progress


def is_terminal(status: str) -> bool:
    return status in TERMINAL


def summarise_config(config: dict[str, Any]) -> tuple[str, int, int]:
    if config.get("agents"):
        return config.get("symbol", ""), 1, len(config["agents"])
    evo = config.get("evolution", {})
    return (config.get("symbol", ""), int(evo.get("generations", 1)),
            int(evo.get("population_size", len(config.get("agents") or []))))
