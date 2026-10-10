from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aether_api.core.config import Settings, get_settings
from aether_api.core.errors import not_found
from aether_api.core.security import require_write_access
from aether_api.database.models import Agent, Experiment, Tournament, TournamentResult, Trade
from aether_api.database.session import get_db
from aether_api.routes.deps import Pagination, paginate
from aether_api.routes.experiments import IdempotencyKey
from aether_api.schemas.common import Page
from aether_api.schemas.tournaments import (
    EquityCurves,
    Leaderboard,
    TournamentCreateRequest,
    TournamentDetail,
    TournamentOut,
    TradeOut,
)
from aether_api.services import experiments as svc
from aether_api.services import queries
from aether_api.services.jobs import enqueue

router = APIRouter(prefix="/tournaments", tags=["tournaments"])


def _get(db: Session, tournament_id: str) -> Tournament:
    t = db.get(Tournament, tournament_id)
    if t is None:
        raise not_found("tournament", tournament_id)
    return t


def _out(db: Session, t: Tournament) -> dict:
    exp = db.get(Experiment, t.experiment_id)
    n = db.scalar(select(func.count()).select_from(TournamentResult)
                  .where(TournamentResult.tournament_id == t.id)) or 0
    return {"id": t.id, "experiment_id": t.experiment_id,
            "experiment_name": exp.name if exp else "", "generation": t.generation,
            "period": t.period, "status": t.status, "config_fingerprint": t.config_fingerprint,
            "n_agents": n, "created_at": t.created_at, "completed_at": t.completed_at}


@router.get("", response_model=Page[TournamentOut])
def list_tournaments(experiment_id: str | None = None, generation: int | None = Query(None, ge=0),
                     page: Pagination = Depends(),
                     db: Session = Depends(get_db)) -> Page[TournamentOut]:
    q = select(Tournament).order_by(Tournament.created_at.desc(), Tournament.generation.desc())
    if experiment_id:
        q = q.where(Tournament.experiment_id == experiment_id)
    if generation is not None:
        q = q.where(Tournament.generation == generation)
    items, total = paginate(db, q, page)
    return Page[TournamentOut](items=[TournamentOut.model_validate(_out(db, t)) for t in items],
                               total=total, limit=page.limit, offset=page.offset)


@router.post("", status_code=status.HTTP_202_ACCEPTED,
             dependencies=[Depends(require_write_access)])
def create_tournament(req: TournamentCreateRequest, response: Response,
                      idempotency_key: str | None = IdempotencyKey,
                      db: Session = Depends(get_db),
                      settings: Settings = Depends(get_settings)) -> dict:
    """Create and queue a single-tournament experiment over an explicit agent list.

    The tournament row appears once the job has evaluated all agents; poll the job.
    """
    exp, created = svc.create_tournament_experiment(db, req, settings, idempotency_key)
    job, job_created = svc.start_experiment(db, exp) if exp.status == "draft" else (
        svc.active_job(db, exp.id), False)
    db.commit()
    if job is not None and job_created:
        enqueue(job.id)
    if not created:
        response.status_code = status.HTTP_200_OK
    return {"experiment_id": exp.id, "job_id": job.id if job else None,
            "status": job.status if job else exp.status}


@router.get("/{tournament_id}", response_model=TournamentDetail)
def get_tournament(tournament_id: str, db: Session = Depends(get_db)) -> TournamentDetail:
    t = _get(db, tournament_id)
    return TournamentDetail.model_validate({**_out(db, t), "evaluation_config": t.evaluation_config,
                                            "folds": t.folds,
                                            "benchmark_metrics": t.benchmark_metrics})


@router.get("/{tournament_id}/leaderboard", response_model=Leaderboard)
def leaderboard(tournament_id: str, db: Session = Depends(get_db)) -> Leaderboard:
    t = _get(db, tournament_id)
    return Leaderboard(tournament_id=t.id, experiment_id=t.experiment_id,
                       generation=t.generation, period="validation",
                       benchmark=t.benchmark_metrics,
                       ranking_policy=t.evaluation_config.get("ranking", {}),
                       rows=queries.leaderboard_rows(db, t))  # type: ignore[arg-type]


@router.get("/{tournament_id}/equity-curves", response_model=EquityCurves)
def equity_curves(tournament_id: str, agent_ids: list[str] | None = Query(None),
                  max_points: int = Query(600, ge=10, le=5000),
                  db: Session = Depends(get_db)) -> EquityCurves:
    t = _get(db, tournament_id)
    data = queries.equity_curves(db, tournament_id=t.id, agent_ids=agent_ids,
                                 max_points=max_points)
    return EquityCurves(tournament_id=t.id, period=t.period, **data)


@router.get("/{tournament_id}/trades", response_model=Page[TradeOut])
def trades(tournament_id: str, agent_id: str | None = None, page: Pagination = Depends(),
           db: Session = Depends(get_db)) -> Page[TradeOut]:
    _get(db, tournament_id)
    q = (select(Trade).join(TournamentResult, TournamentResult.id == Trade.tournament_result_id)
         .where(TournamentResult.tournament_id == tournament_id)
         .order_by(Trade.timestamp, Trade.id))
    if agent_id:
        q = q.where(TournamentResult.agent_id == agent_id)
    items, total = paginate(db, q, page)
    owners = dict(db.execute(select(TournamentResult.id, TournamentResult.agent_id).where(
        TournamentResult.tournament_id == tournament_id)).all())
    out = [TradeOut.model_validate({**{c: getattr(tr, c) for c in TradeOut.model_fields
                                       if c != "agent_id"},
                                    "agent_id": owners[tr.tournament_result_id]})
           for tr in items]
    return Page[TradeOut](items=out, total=total, limit=page.limit, offset=page.offset)


__all__ = ["Agent", "router"]
