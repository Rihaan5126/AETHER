"""Durable job records and execution backends.

The ``jobs`` table is the source of truth for job state. Two backends execute the
same :func:`run_job` function:

* ``inline``: a bounded in-process thread pool (tests and laptop development).
* ``celery``: a Redis-backed Celery queue consumed by the separate worker
  service (``aether_api.worker``), which is how Docker Compose and deployments
  run training outside the HTTP process.

States: queued -> running -> completed | failed | cancelled. Cancellation is
cooperative: the job checks ``cancel_requested`` between units of work. A job
that exceeds ``max_job_seconds`` fails with a timeout. On API/worker start,
jobs left ``running`` by a dead process (stale heartbeat) are marked failed with
an explicit "interrupted" message, never silently resumed.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from aether_api.core.config import get_settings
from aether_api.core.logging import get_logger
from aether_api.database.models import Job, new_id, utcnow
from aether_api.database.session import session_scope

log = get_logger(__name__)
TERMINAL = {"completed", "failed", "cancelled"}
HEARTBEAT_STALE = timedelta(minutes=10)


class JobCancelled(Exception):  # noqa: N818
    pass


class JobTimeout(Exception):  # noqa: N818
    pass


class JobContext:
    """Handle given to job functions for progress reporting and cancellation."""

    def __init__(self, job_id: str, max_seconds: int) -> None:
        self.job_id = job_id
        self.started = time.monotonic()
        self.max_seconds = max_seconds
        self._last_write = 0.0

    def progress(self, fraction: float, message: str, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_write < 0.5:
            return
        self._last_write = now
        with session_scope() as db:
            db.execute(
                update(Job)
                .where(Job.id == self.job_id)
                .values(
                    progress=max(0.0, min(1.0, fraction)),
                    message=message[:500],
                    heartbeat_at=utcnow(),
                )
            )

    def should_cancel(self) -> bool:
        if time.monotonic() - self.started > self.max_seconds:
            raise JobTimeout(f"job exceeded max duration of {self.max_seconds}s")
        with session_scope() as db:
            flag = db.execute(select(Job.cancel_requested).where(Job.id == self.job_id)).scalar()
        return bool(flag)

    def check(self) -> None:
        if self.should_cancel():
            raise JobCancelled("cancelled by user")


JobHandler = Callable[[JobContext, dict[str, Any]], dict[str, Any]]
FailureHook = Callable[[Session, dict[str, Any], str, str | None], None]
_HANDLERS: dict[str, JobHandler] = {}
_FAILURE_HOOKS: dict[str, FailureHook] = {}


def register(
    job_type: str, on_failure: FailureHook | None = None
) -> Callable[[JobHandler], JobHandler]:
    """Register a handler; ``on_failure(db, payload, status, error)`` marks related records."""

    def deco(fn: JobHandler) -> JobHandler:
        _HANDLERS[job_type] = fn
        if on_failure is not None:
            _FAILURE_HOOKS[job_type] = on_failure
        return fn

    return deco


def create_job(
    db: Session, job_type: str, payload: dict[str, Any], experiment_id: str | None = None
) -> Job:
    if job_type not in _HANDLERS:
        raise ValueError(f"unknown job type {job_type}")
    job = Job(
        id=new_id("job"),
        job_type=job_type,
        status="queued",
        payload=payload,
        experiment_id=experiment_id,
        message="queued",
    )
    db.add(job)
    db.flush()
    return job


def run_job(job_id: str) -> None:
    """Execute a queued job. Safe to call from a thread or a Celery worker."""
    settings = get_settings()
    with session_scope() as db:
        job = db.get(Job, job_id)
        if job is None:
            log.error("job.missing", job_id=job_id)
            return
        if job.status in TERMINAL:
            return
        if job.cancel_requested:
            _finish(db, job, "cancelled", message="cancelled before start")
            _on_failure(db, job, "cancelled", None)
            return
        job.status, job.started_at, job.heartbeat_at = "running", utcnow(), utcnow()
        job.attempts += 1
        job_type, payload = job.job_type, dict(job.payload)
    ctx = JobContext(job_id, settings.max_job_seconds)
    log.info("job.start", job_id=job_id, job_type=job_type)
    try:
        result = _HANDLERS[job_type](ctx, payload)
        status, error, message = "completed", None, "completed"
    except JobCancelled as exc:
        result, status, error, message = None, "cancelled", None, str(exc)
    except JobTimeout as exc:
        result, status, error, message = None, "failed", str(exc), "timed out"
    except Exception as exc:
        log.error("job.failed", job_id=job_id, error=str(exc), tb=traceback.format_exc())
        result, status, error, message = None, "failed", f"{type(exc).__name__}: {exc}", "failed"
    with session_scope() as db:
        job = db.get(Job, job_id)
        assert job is not None
        _finish(db, job, status, result=result, error=error, message=message)
        if status != "completed":
            _on_failure(db, job, status, error)
    log.info("job.end", job_id=job_id, status=status)


def _finish(
    db: Session,
    job: Job,
    status: str,
    *,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    message: str = "",
) -> None:
    job.status = status
    job.result = result
    job.error_message = error
    job.message = message
    job.completed_at = utcnow()
    if status == "completed":
        job.progress = 1.0


def _on_failure(db: Session, job: Job, status: str, error: str | None) -> None:
    hook = _FAILURE_HOOKS.get(job.job_type)
    if hook is not None:
        hook(db, dict(job.payload), status, error)


def request_cancel(db: Session, job: Job) -> None:
    if job.status in TERMINAL:
        return
    job.cancel_requested = True
    if job.status == "queued":
        _finish(db, job, "cancelled", message="cancelled before start")
        _on_failure(db, job, "cancelled", None)


def recover_interrupted_jobs() -> int:
    """Mark jobs orphaned by a crashed process as failed."""
    cutoff = datetime.now(UTC) - HEARTBEAT_STALE
    count = 0
    with session_scope() as db:
        for job in db.scalars(select(Job).where(Job.status == "running")):
            beat = job.heartbeat_at or job.started_at
            if beat is not None and beat.tzinfo is None:
                beat = beat.replace(tzinfo=UTC)
            if beat is None or beat < cutoff:
                error = "interrupted: worker stopped while running"
                _finish(db, job, "failed", error=error, message="interrupted")
                _on_failure(db, job, "failed", error)
                count += 1
    return count


# ---- backends ------------------------------------------------------------------------
class InlineBackend:
    def __init__(self, max_workers: int) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="aether-job")
        self._lock = threading.Lock()

    def submit(self, job_id: str) -> None:
        self._pool.submit(run_job, job_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class CeleryBackend:
    def submit(self, job_id: str) -> None:
        from aether_api.worker import celery_app  # noqa: PLC0415

        celery_app.send_task("aether.run_job", args=[job_id], queue="training")


_backend: InlineBackend | CeleryBackend | None = None


def get_backend() -> InlineBackend | CeleryBackend:
    global _backend  # noqa: PLW0603
    if _backend is None:
        settings = get_settings()
        _backend = (
            CeleryBackend()
            if settings.job_backend == "celery"
            else InlineBackend(settings.max_concurrent_jobs)
        )
    return _backend


def reset_backend() -> None:
    global _backend  # noqa: PLW0603
    if isinstance(_backend, InlineBackend):
        _backend.shutdown()
    _backend = None


def enqueue(job_id: str) -> None:
    get_backend().submit(job_id)
