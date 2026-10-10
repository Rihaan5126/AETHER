from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from aether_api.core.errors import not_found
from aether_api.database.models import Job
from aether_api.database.session import get_db
from aether_api.routes.deps import Pagination, paginate
from aether_api.schemas.common import Page
from aether_api.schemas.jobs import JobOut

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=Page[JobOut])
def list_jobs(status: str | None = Query(None, pattern="^(queued|running|completed|failed|cancelled)$"),
              experiment_id: str | None = None, page: Pagination = Depends(),
              db: Session = Depends(get_db)) -> Page[JobOut]:
    q = select(Job).order_by(Job.created_at.desc())
    if status:
        q = q.where(Job.status == status)
    if experiment_id:
        q = q.where(Job.experiment_id == experiment_id)
    items, total = paginate(db, q, page)
    return Page[JobOut](items=[JobOut.model_validate(j) for j in items], total=total,
                        limit=page.limit, offset=page.offset)


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobOut:
    job = db.get(Job, job_id)
    if job is None:
        raise not_found("job", job_id)
    return JobOut.model_validate(job)
