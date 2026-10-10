"""Background job handlers that run research workloads and persist their results."""

from __future__ import annotations

import resource
import time
from typing import Any, cast

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from trading_engine.agents import load_agent, parse_genome, save_agent
from trading_engine.agents.genomes import BENCHMARK_FAMILIES
from trading_engine.evolution import GenerationResult, LineageRecord, run_evolution
from trading_engine.experiments import (
    ExperimentConfig,
    build_manifest,
    fit_on_development,
    prepare_market_data,
    run_final_test,
)
from trading_engine.experiments.final_test import BENCHMARK_ID
from trading_engine.optimization import OptimizationConfig, optimize
from trading_engine.storage import ArtifactRef
from trading_engine.tournaments import AgentEvaluation, AgentSpec, run_tournament
from trading_engine.tournaments.engine import CancelledError

from aether_api.core.config import get_settings
from aether_api.database.models import (
    Agent,
    EquityCurve,
    Experiment,
    FinalTestRun,
    ModelLineage,
    OptimizationStudy,
    OptimizationTrial,
    Tournament,
    TournamentResult,
    Trade,
    TrainingRun,
    new_id,
    utcnow,
)
from aether_api.database.session import session_scope
from aether_api.services.jobs import JobCancelled, JobContext, register
from aether_api.services.storage import (
    artifact_store,
    dataset_store,
    write_parquet_artifact,
)

FAMILY_LABELS = {
    "random": "Random",
    "buy_and_hold": "Buy & Hold",
    "momentum": "Momentum",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "mlp": "MLP",
}
BENCHMARK_CURVE_ID = "__benchmark__"


def agent_name(family: str, generation: int, index: int) -> str:
    return f"{FAMILY_LABELS.get(family, family)} G{generation}·{index:02d}"


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _usage() -> dict[str, float]:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "cpu_user_seconds": round(ru.ru_utime, 2),
        "cpu_system_seconds": round(ru.ru_stime, 2),
        "max_rss_mb": round(ru.ru_maxrss / 1024, 1),
    }


def _experiment(db: Session, exp_id: str) -> Experiment:
    exp = db.get(Experiment, exp_id)
    if exp is None:
        raise LookupError(f"experiment {exp_id} not found")
    return exp


# ---- persistence ------------------------------------------------------------------------
def persist_generation(
    db: Session, exp: Experiment, config: ExperimentConfig, gen: GenerationResult
) -> Tournament:
    lineage_by_id: dict[str, LineageRecord] = {rec.child_id: rec for rec in gen.lineage}
    for i, spec in enumerate(gen.specs):
        if db.get(Agent, spec.agent_id) is not None:
            continue
        rec = lineage_by_id[spec.agent_id]
        family = spec.family
        db.add(
            Agent(
                id=spec.agent_id,
                experiment_id=exp.id,
                name=agent_name(family, gen.generation, i),
                algorithm=family,
                genome=rec.genome_after,
                generation=gen.generation,
                parent_id=rec.parent_ids[0] if rec.parent_ids else None,
                origin=rec.operation,
                seed=spec.seed,
                is_benchmark=family in BENCHMARK_FAMILIES,
            )
        )
    db.flush()
    for rec in gen.lineage:
        parents: list[str | None] = [*rec.parent_ids] or [None]
        for parent in parents:
            db.add(
                ModelLineage(
                    experiment_id=exp.id,
                    child_id=rec.child_id,
                    parent_id=parent,
                    generation=rec.generation,
                    operation=rec.operation,
                    mutation_description=rec.mutations,
                    genome_before=rec.genome_before,
                    genome_after=rec.genome_after,
                    training_seed=rec.seed,
                    validation_score=rec.validation_score,
                )
            )
    outcome = gen.outcome
    bh = next((e for e in outcome.evaluations if e.family == "buy_and_hold"), None)
    tournament = Tournament(
        id=new_id("trn"),
        experiment_id=exp.id,
        generation=gen.generation,
        period="validation",
        evaluation_config=config.evaluation.model_dump(mode="json"),
        config_fingerprint=outcome.config_fingerprint,
        folds=outcome.folds,
        benchmark_metrics=bh.metrics if bh else None,
        status="completed",
        completed_at=utcnow(),
    )
    db.add(tournament)
    db.flush()
    persist_results(
        db, tournament, exp, config.symbol.upper(), outcome.evaluations, outcome.benchmark_equity
    )
    return tournament


def persist_results(
    db: Session,
    tournament: Tournament,
    exp: Experiment,
    symbol: str,
    evaluations: list[AgentEvaluation],
    benchmark_equity: pd.DataFrame,
) -> None:
    curves = []
    for ev in evaluations:
        result = TournamentResult(
            id=new_id("res"),
            tournament_id=tournament.id,
            agent_id=ev.agent_id,
            total_return=_num(ev.metrics.get("total_return")),
            sharpe_ratio=_num(ev.metrics.get("sharpe_ratio")),
            max_drawdown=_num(ev.metrics.get("max_drawdown")),
            score=ev.score,
            rank=ev.rank,
            status=ev.status,
            status_reason=ev.status_reason,
            metrics_json=ev.metrics,
            fold_metrics=ev.fold_metrics,
            score_components=ev.score_components,
            warnings=ev.warnings,
            risk_events=ev.risk_events[:200],
            evaluation_seconds=ev.wall_seconds,
        )
        db.add(result)
        db.add(
            TrainingRun(
                id=new_id("run"),
                experiment_id=exp.id,
                agent_id=ev.agent_id,
                tournament_id=tournament.id,
                status="failed" if ev.error else "completed",
                started_at=utcnow(),
                completed_at=utcnow(),
                training_metrics={"folds": ev.fold_training, "cached_from": ev.cached_from},
                error_message=ev.error,
            )
        )
        db.flush()
        if ev.trades is not None and len(ev.trades):
            db.add_all(
                [
                    Trade(
                        tournament_result_id=result.id,
                        symbol=symbol,
                        timestamp=pd.Timestamp(t.executed_time).to_pydatetime(),
                        signal_time=pd.Timestamp(t.signal_time).to_pydatetime(),
                        side=t.side,
                        quantity=float(t.quantity),
                        execution_price=float(t.fill_price),
                        reference_price=float(t.reference_price),
                        fees=float(t.fee),
                        slippage=float(t.slippage_cost),
                        notional=float(t.notional),
                        target_weight=float(t.target_weight),
                    )
                    for t in cast(Any, ev.trades).itertuples(index=False)
                ]
            )
        if ev.equity is not None:
            curves.append(_curve_frame(ev.agent_id, ev.equity))
    curves.append(_curve_frame(BENCHMARK_CURVE_ID, benchmark_equity))
    frame = pd.concat(curves, ignore_index=True)
    rel, digest = write_parquet_artifact(
        frame, "experiments", exp.id, f"tournament-{tournament.generation}-equity.parquet"
    )
    db.add(
        EquityCurve(
            id=new_id("eq"), tournament_id=tournament.id, path=rel, sha256=digest, rows=len(frame)
        )
    )


def _curve_frame(agent_id: str, equity: pd.DataFrame) -> pd.DataFrame:
    out = equity.reset_index()[["timestamp", "equity", "drawdown", "weight"]].copy()
    out["fold"] = equity["fold"].to_numpy() if "fold" in equity.columns else 0
    out.insert(0, "agent_id", agent_id)
    return out


def load_experiment(db: Session, experiment_id: str) -> tuple[Experiment, ExperimentConfig]:
    exp = db.get(Experiment, experiment_id)
    if exp is None:
        raise RuntimeError(f"experiment {experiment_id} not found")
    return exp, ExperimentConfig.model_validate(exp.config)


# ---- failure hooks ----------------------------------------------------------------------
def _experiment_failed(
    db: Session, payload: dict[str, Any], status: str, error: str | None
) -> None:
    exp = db.get(Experiment, payload["experiment_id"])
    if exp is not None and exp.status not in {"completed", "failed", "cancelled"}:
        exp.status, exp.error_message, exp.completed_at = status, error, utcnow()


def _final_test_failed(
    db: Session, payload: dict[str, Any], status: str, error: str | None
) -> None:
    run = db.get(FinalTestRun, payload["final_test_run_id"])
    if run is not None and run.status != "completed":
        run.status, run.completed_at = status, utcnow()
        run.results = [{"error": error}] if error else []


def _study_failed(db: Session, payload: dict[str, Any], status: str, error: str | None) -> None:
    study = db.get(OptimizationStudy, payload["study_id"])
    if study is not None and study.status != "completed":
        study.status, study.completed_at = status, utcnow()


# ---- experiment job ---------------------------------------------------------------------
@register("run_experiment", on_failure=_experiment_failed)
def run_experiment_job(ctx: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    exp_id = payload["experiment_id"]
    settings = get_settings()
    with session_scope() as db:
        exp, config = load_experiment(db, exp_id)
        exp.status, exp.started_at = "running", utcnow()
        if exp.manifest is None:
            exp.manifest = build_manifest(exp.id, config)
            exp.code_version = exp.manifest["code_version"]
        kind = exp.kind
    ctx.progress(0.01, "loading dataset and features", force=True)
    dev, test = prepare_market_data(dataset_store(), config)
    with session_scope() as db:
        exp = _experiment(db, exp_id)
        exp.development_period = {
            "start": dev.start.date().isoformat(),
            "end": dev.end.date().isoformat(),
            "sessions": len(dev.index),
        }
        exp.test_period = {
            "start": test.start.date().isoformat(),
            "end": test.end.date().isoformat(),
            "sessions": len(test.index),
            "locked": True,
        }
    workers = min(config.max_workers, settings.max_workers_per_job)
    try:
        if kind == "tournament":
            last_specs = _run_single_tournament(ctx, exp_id, config, dev, workers)
        else:
            last_specs = _run_evolution(ctx, exp_id, config, dev, workers)
    except CancelledError as exc:
        raise JobCancelled(str(exc)) from exc
    ctx.check()
    _save_final_artifacts(ctx, exp_id, config, dev, last_specs)
    with session_scope() as db:
        exp = _experiment(db, exp_id)
        exp.status, exp.completed_at = "completed", utcnow()
        exp.wall_seconds = round(time.perf_counter() - started, 2)
        exp.resource_usage = _usage()
        n_agents = db.scalar(
            select(func.count()).select_from(Agent).where(Agent.experiment_id == exp_id)
        )
    return {"experiment_id": exp_id, "agents": n_agents}


def _run_evolution(
    ctx: JobContext, exp_id: str, config: ExperimentConfig, dev: Any, workers: int
) -> list[AgentSpec]:
    total_gens = config.evolution.generations

    def on_agent(gen: int, done: int, total: int, ev: AgentEvaluation) -> None:
        frac = 0.02 + 0.83 * (gen + done / max(total, 1)) / total_gens
        ctx.progress(
            frac,
            f"generation {gen}: evaluated {done}/{total} new agents "
            f"({ev.agent_id.split('-', 1)[-1]})",
        )

    def on_generation(gen: GenerationResult) -> None:
        with session_scope() as db:
            exp, _ = load_experiment(db, exp_id)
            persist_generation(db, exp, config, gen)
            exp.current_generation = gen.generation
        ctx.progress(
            0.02 + 0.83 * (gen.generation + 1) / total_gens,
            f"generation {gen.generation} tournament stored",
            force=True,
        )

    result = run_evolution(
        dev,
        config.evolution,
        config.evaluation,
        seed=config.seed,
        id_factory=lambda g, i, fam: f"{exp_id}-g{g}-{i:02d}-{fam}",
        max_workers=workers,
        on_generation=on_generation,
        on_agent=on_agent,
        should_cancel=ctx.should_cancel,
    )
    if len(result.generations) < total_gens:
        ctx.check()
    return result.generations[-1].specs if result.generations else []


def _run_single_tournament(
    ctx: JobContext, exp_id: str, config: ExperimentConfig, dev: Any, workers: int
) -> list[AgentSpec]:
    specs = []
    for i, choice in enumerate(config.agents or []):
        genome = parse_genome(choice.get("genome") or {"family": choice["family"]})
        seed = choice.get("seed")
        specs.append(
            AgentSpec(
                f"{exp_id}-g0-{i:02d}-{genome.family}",
                genome,
                int(seed) if seed is not None else config.seed + i,
            )
        )

    def progress(done: int, total: int, ev: AgentEvaluation) -> None:
        ctx.progress(0.02 + 0.83 * done / total, f"evaluated {done}/{total} agents")

    outcome = run_tournament(
        specs,
        dev,
        config.evaluation,
        max_workers=workers,
        on_progress=progress,
        should_cancel=ctx.should_cancel,
    )
    lineage = [
        LineageRecord(
            s.agent_id, 0, "initial", [], [], None, s.genome.model_dump(mode="json"), s.seed
        )
        for s in specs
    ]
    score = {e.agent_id: e.score for e in outcome.evaluations}
    for rec in lineage:
        rec.validation_score = score.get(rec.child_id)
    with session_scope() as db:
        exp, _ = load_experiment(db, exp_id)
        persist_generation(db, exp, config, GenerationResult(0, specs, outcome, lineage))
        exp.current_generation = 0
    return specs


def _save_final_artifacts(
    ctx: JobContext, exp_id: str, config: ExperimentConfig, dev: Any, specs: list[AgentSpec]
) -> None:
    """Fit the final population on all development data and store verified artifacts."""
    store = artifact_store()
    for i, spec in enumerate(specs):
        ctx.check()
        ctx.progress(
            0.85 + 0.14 * i / max(len(specs), 1),
            f"fitting final model {i + 1}/{len(specs)} on development data",
        )
        agent = fit_on_development(
            spec,
            dev,
            horizon=config.evaluation.walk_forward.horizon,
            max_seconds=config.evaluation.max_fit_seconds,
        )
        ref = save_agent(agent, store, f"experiments/{exp_id}/agents/{spec.agent_id}")
        with session_scope() as db:
            row = db.get(Agent, spec.agent_id)
            if row is not None:
                row.artifact_path, row.artifact_sha256 = ref.key, ref.sha256
                row.artifact_files = ref.files


# ---- final test job ---------------------------------------------------------------------
@register("final_test", on_failure=_final_test_failed)
def final_test_job(ctx: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    run_id = payload["final_test_run_id"]
    with session_scope() as db:
        run = db.get(FinalTestRun, run_id)
        assert run is not None
        exp, config = load_experiment(db, run.experiment_id)
        refs = []
        for agent_id in run.agent_ids:
            agent = db.get(Agent, agent_id)
            if agent is None or not agent.artifact_path or agent.artifact_files is None:
                raise RuntimeError(f"agent {agent_id} has no stored model artifact")
            refs.append(
                ArtifactRef(
                    agent.artifact_path, agent.artifact_sha256 or "", dict(agent.artifact_files)
                )
            )
    ctx.progress(0.1, "loading verified model artifacts", force=True)
    store = artifact_store()
    agents = [load_agent(store, ref) for ref in refs]
    dev, test = prepare_market_data(dataset_store(), config)
    ctx.progress(0.4, "simulating the locked test period once", force=True)
    result = run_final_test(agents, dev, test, config.evaluation)
    frames = [
        _curve_frame(aid if aid != BENCHMARK_ID else BENCHMARK_CURVE_ID, eq)
        for aid, eq in result.equity.items()
    ]
    rel, digest = write_parquet_artifact(
        pd.concat(frames, ignore_index=True),
        "experiments",
        exp.id,
        f"final-test-{run.run_number}-equity.parquet",
    )
    with session_scope() as db:
        run = db.get(FinalTestRun, run_id)
        assert run is not None
        run.status, run.completed_at = "completed", utcnow()
        run.results = result.agents
        run.benchmark_metrics = result.benchmark_metrics
        run.test_period = {"start": result.test_start, "end": result.test_end}
        db.add(
            EquityCurve(
                id=new_id("eq"),
                final_test_run_id=run.id,
                path=rel,
                sha256=digest,
                rows=sum(len(f) for f in frames),
            )
        )
    return {"final_test_run_id": run_id, "agents": len(agents)}


# ---- Optuna job -------------------------------------------------------------------------
@register("optimize", on_failure=_study_failed)
def optimize_job(ctx: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    study_id = payload["study_id"]
    with session_scope() as db:
        study = db.get(OptimizationStudy, study_id)
        assert study is not None
        _, config = load_experiment(db, study.experiment_id)
        opt_cfg = OptimizationConfig.model_validate(study.config)
    dev, _ = prepare_market_data(dataset_store(), config)
    done = {"n": 0}

    def on_trial(rec: Any) -> None:
        done["n"] += 1
        with session_scope() as db:
            db.add(
                OptimizationTrial(
                    study_id=study_id,
                    number=rec.number,
                    state=rec.state,
                    params=rec.params,
                    value=rec.value,
                    fold_scores=rec.fold_scores,
                    duration_seconds=rec.duration_seconds,
                    error=rec.error,
                )
            )
        ctx.progress(
            done["n"] / opt_cfg.n_trials, f"trial {done['n']}/{opt_cfg.n_trials}", force=True
        )

    result = optimize(
        dev, config.evaluation, opt_cfg, on_trial=on_trial, should_cancel=ctx.should_cancel
    )
    with session_scope() as db:
        study = db.get(OptimizationStudy, study_id)
        assert study is not None
        recorded = {t.number for t in study.trials}
        for rec in result.trials:
            if rec.number not in recorded:
                db.add(
                    OptimizationTrial(
                        study_id=study_id,
                        number=rec.number,
                        state=rec.state,
                        params=rec.params,
                        value=rec.value,
                        fold_scores=rec.fold_scores,
                        duration_seconds=rec.duration_seconds,
                        error=rec.error,
                    )
                )
        study.status, study.completed_at = "completed", utcnow()
        study.best_value, study.best_params = result.best_value, result.best_params
        study.best_genome = result.best_genome
    ctx.check()
    return {"study_id": study_id, "trials": len(result.trials), "best_value": result.best_value}
