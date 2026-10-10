from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, Header, Query, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from trading_engine.experiments import ExperimentConfig, estimate_workload

from aether_api.core.config import Settings, get_settings
from aether_api.core.errors import not_found
from aether_api.core.security import require_write_access
from aether_api.database.models import (
    Experiment,
    FinalTestRun,
    Job,
    OptimizationStudy,
    Tournament,
    TournamentResult,
)
from aether_api.database.session import get_db
from aether_api.routes.deps import Pagination, paginate
from aether_api.schemas.agents import LineageGraph
from aether_api.schemas.common import JobRef, Page
from aether_api.schemas.experiments import (
    ExperimentCreateRequest,
    ExperimentDetail,
    ExperimentOut,
    FinalTestRequest,
    GenerationSummary,
    OptimizeRequest,
)
from aether_api.schemas.jobs import JobOut
from aether_api.services import experiments as svc
from aether_api.services import queries
from aether_api.services.jobs import enqueue

router = APIRouter(prefix="/experiments", tags=["experiments"])
IdempotencyKey = Header(default=None, alias="Idempotency-Key", max_length=128)


def _get(db: Session, experiment_id: str) -> Experiment:
    exp = db.get(Experiment, experiment_id)
    if exp is None:
        raise not_found("experiment", experiment_id)
    return exp


def to_out(db: Session, exp: Experiment) -> dict[str, Any]:
    symbol, gens, pop = svc.summarise_config(exp.config)
    return {
        "id": exp.id,
        "name": exp.name,
        "kind": exp.kind,
        "status": exp.status,
        "dataset_id": exp.dataset_id,
        "symbol": symbol,
        "seed": exp.seed,
        "current_generation": exp.current_generation,
        "planned_generations": gens,
        "population_size": pop,
        "progress": svc.latest_job_progress(db, exp),
        "test_contaminated": exp.test_contaminated,
        "created_at": exp.created_at,
        "started_at": exp.started_at,
        "completed_at": exp.completed_at,
    }


def generation_summaries(db: Session, exp: Experiment) -> list[GenerationSummary]:
    out = []
    for t in db.scalars(
        select(Tournament).where(Tournament.experiment_id == exp.id).order_by(Tournament.generation)
    ):
        rows = db.scalars(
            select(TournamentResult)
            .where(TournamentResult.tournament_id == t.id)
            .order_by(TournamentResult.rank)
        ).all()
        active = [r for r in rows if not r.agent.is_benchmark and r.score is not None]
        eligible = [r for r in rows if r.status == "eligible"]
        rand = next((r for r in rows if r.agent.algorithm == "random"), None)
        bh = next((r for r in rows if r.agent.algorithm == "buy_and_hold"), None)
        out.append(
            GenerationSummary(
                generation=t.generation,
                tournament_id=t.id,
                status=t.status,
                n_agents=len(rows),
                best_score=max((r.score for r in active), default=None),  # type: ignore[type-var]
                median_score=float(np.median([r.score for r in active])) if active else None,
                best_agent_id=eligible[0].agent_id if eligible else None,
                random_baseline_score=rand.score if rand else None,
                buy_and_hold_return=bh.total_return if bh else None,
            )
        )
    return out


@router.get("", response_model=Page[ExperimentOut])
def list_experiments(
    page: Pagination = Depends(),
    status_filter: str | None = Query(None, alias="status"),
    kind: str | None = Query(None, pattern="^(evolution|tournament)$"),
    db: Session = Depends(get_db),
) -> Page[ExperimentOut]:
    q = select(Experiment).order_by(Experiment.created_at.desc())
    if status_filter:
        q = q.where(Experiment.status == status_filter)
    if kind:
        q = q.where(Experiment.kind == kind)
    items, total = paginate(db, q, page)
    return Page[ExperimentOut](
        items=[ExperimentOut.model_validate(to_out(db, e)) for e in items],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.post(
    "",
    response_model=ExperimentOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_write_access)],
)
def create_experiment(
    req: ExperimentCreateRequest,
    response: Response,
    idempotency_key: str | None = IdempotencyKey,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ExperimentOut:
    exp, created = svc.create_experiment(db, req, settings, idempotency_key)
    db.commit()
    if not created:
        response.status_code = status.HTTP_200_OK
    return ExperimentOut.model_validate(to_out(db, exp))


@router.post("/estimate")
def estimate(
    req: ExperimentCreateRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Relative workload for a proposed experiment (no runtime estimate)."""
    ds = svc._dataset(db, req.dataset_id, req.symbol)
    return estimate_workload(svc.build_config(req, ds, settings))


@router.get("/{experiment_id}", response_model=ExperimentDetail)
def get_experiment(experiment_id: str, db: Session = Depends(get_db)) -> ExperimentDetail:
    exp = _get(db, experiment_id)
    jobs = db.scalars(
        select(Job).where(Job.experiment_id == exp.id).order_by(Job.created_at.desc())
    ).all()
    runs = db.scalars(
        select(FinalTestRun)
        .where(FinalTestRun.experiment_id == exp.id)
        .order_by(FinalTestRun.run_number)
    ).all()
    studies = db.scalars(
        select(OptimizationStudy).where(OptimizationStudy.experiment_id == exp.id)
    ).all()
    return ExperimentDetail.model_validate(
        {
            **to_out(db, exp),
            "config": exp.config,
            "manifest": exp.manifest,
            "code_version": exp.code_version,
            "development_period": exp.development_period,
            "test_period": exp.test_period,
            "error_message": exp.error_message,
            "wall_seconds": exp.wall_seconds,
            "resource_usage": exp.resource_usage,
            "workload": estimate_workload(ExperimentConfig.model_validate(exp.config)),
            "generations": generation_summaries(db, exp),
            "jobs": [JobOut.model_validate(j).model_dump(mode="json") for j in jobs],
            "final_test_runs": [
                {
                    "id": r.id,
                    "run_number": r.run_number,
                    "status": r.status,
                    "agent_ids": r.agent_ids,
                    "results": r.results,
                    "benchmark_metrics": r.benchmark_metrics,
                    "test_period": r.test_period,
                    "acknowledged_contamination": r.acknowledged_contamination,
                    "created_at": r.created_at.isoformat(),
                }
                for r in runs
            ],
            "optimization_studies": [
                {
                    "id": s.id,
                    "family": s.family,
                    "status": s.status,
                    "best_value": s.best_value,
                    "best_params": s.best_params,
                    "n_trials": len(s.trials),
                }
                for s in studies
            ],
        }
    )


@router.post(
    "/{experiment_id}/start",
    response_model=JobRef,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_write_access)],
)
def start_experiment(
    experiment_id: str, response: Response, db: Session = Depends(get_db)
) -> JobRef:
    exp = _get(db, experiment_id)
    job, created = svc.start_experiment(db, exp)
    db.commit()
    if created:
        enqueue(job.id)
    else:
        response.status_code = status.HTTP_200_OK
    return JobRef(job_id=job.id, status=job.status)


@router.post(
    "/{experiment_id}/cancel",
    response_model=ExperimentOut,
    dependencies=[Depends(require_write_access)],
)
def cancel_experiment(experiment_id: str, db: Session = Depends(get_db)) -> ExperimentOut:
    exp = _get(db, experiment_id)
    svc.cancel_experiment(db, exp)
    db.commit()
    db.refresh(exp)
    return ExperimentOut.model_validate(to_out(db, exp))


@router.post(
    "/{experiment_id}/final-test",
    response_model=JobRef,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_write_access)],
)
def final_test(experiment_id: str, req: FinalTestRequest, db: Session = Depends(get_db)) -> JobRef:
    exp = _get(db, experiment_id)
    _, job = svc.request_final_test(db, exp, req)
    db.commit()
    enqueue(job.id)
    return JobRef(job_id=job.id, status=job.status)


@router.get("/{experiment_id}/final-test/{run_number}/equity-curves")
def final_test_curves(
    experiment_id: str,
    run_number: int,
    max_points: int = Query(600, ge=10, le=5000),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    run = db.scalar(
        select(FinalTestRun).where(
            FinalTestRun.experiment_id == experiment_id, FinalTestRun.run_number == run_number
        )
    )
    if run is None:
        raise not_found("final test run", run_number)
    return {
        "run_number": run_number,
        "period": "test",
        **queries.equity_curves(db, final_test_run_id=run.id, max_points=max_points),
    }


@router.post(
    "/{experiment_id}/optimize",
    response_model=JobRef,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_write_access)],
)
def optimize(
    experiment_id: str,
    req: OptimizeRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> JobRef:
    exp = _get(db, experiment_id)
    _, job = svc.request_optimization(db, exp, req, settings)
    db.commit()
    enqueue(job.id)
    return JobRef(job_id=job.id, status=job.status)


@router.get("/{experiment_id}/lineage", response_model=LineageGraph)
def experiment_lineage(experiment_id: str, db: Session = Depends(get_db)) -> LineageGraph:
    _get(db, experiment_id)
    return LineageGraph.model_validate(queries.lineage_graph(db, experiment_id))


@router.get("/{experiment_id}/analytics")
def experiment_analytics(experiment_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    return queries.experiment_analytics(db, _get(db, experiment_id))
