"""Celery worker entry point (separately deployable training worker).

Run with::

    celery -A aether_api.worker worker -Q training --concurrency=1 --loglevel=INFO

Concurrency defaults to 1 because each job is CPU-bound; scale by adding worker
containers rather than threads.
"""

from __future__ import annotations

from celery import Celery
from celery.signals import worker_ready
from sqlalchemy.exc import OperationalError

from aether_api.core.config import get_settings
from aether_api.core.logging import configure_logging
from aether_api.services import datasets as _datasets  # noqa: F401  (register handlers)
from aether_api.services import runner as _runner  # noqa: F401
from aether_api.services.jobs import recover_interrupted_jobs, run_job

settings = get_settings()
configure_logging(settings.log_level, settings.log_json)

celery_app = Celery("aether", broker=settings.redis_url, backend=None)
celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=False,
    task_time_limit=settings.max_job_seconds + 300,
    task_soft_time_limit=settings.max_job_seconds + 60,
    task_default_queue="training",
    broker_connection_retry_on_startup=True,
)


@celery_app.task(name="aether.run_job", autoretry_for=(OperationalError,), retry_backoff=True,
                 max_retries=3)
def run_job_task(job_id: str) -> None:
    run_job(job_id)


@worker_ready.connect
def _recover(**_: object) -> None:
    recover_interrupted_jobs()
