from __future__ import annotations

import os
import shutil

import psutil
from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session
from trading_engine.experiments import code_version, dependency_versions

from aether_api import __version__
from aether_api.core.config import Settings, get_settings
from aether_api.database.models import Job
from aether_api.database.session import get_db
from aether_api.schemas.system import HealthOut, SystemStatus

router = APIRouter(tags=["system"])

DISCLAIMER = (
    "AETHER is a research platform. All trading is simulated on historical or synthetic data; "
    "no orders are ever sent to a broker. Backtest results are not evidence of future "
    "profitability and nothing here is investment advice."
)


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(status="ok", version=__version__)


@router.get("/system/status", response_model=SystemStatus)
def system_status(
    db: Session = Depends(get_db), settings: Settings = Depends(get_settings)
) -> SystemStatus:
    try:
        db.execute(text("SELECT 1"))
        database = {"ok": True, "dialect": db.get_bind().dialect.name}
    except Exception as exc:  # reported to the client, not swallowed
        database = {"ok": False, "error": str(exc)}
    queue: dict[str, object] = {"backend": settings.job_backend}
    if settings.job_backend == "celery":
        try:
            import redis  # noqa: PLC0415

            client = redis.Redis.from_url(settings.redis_url, socket_timeout=1)
            queue["redis_ok"] = bool(client.ping())
            queue["pending"] = int(client.llen("training"))
        except Exception as exc:
            queue["redis_ok"] = False
            queue["error"] = str(exc)
    counts = dict(db.execute(select(Job.status, func.count()).group_by(Job.status)).all())
    jobs = {
        s: int(counts.get(s, 0)) for s in ("queued", "running", "completed", "failed", "cancelled")
    }
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(settings.artifact_dir if settings.artifact_dir.exists() else "/")
    resources = {
        "cpu_count": os.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_total_mb": round(vm.total / 2**20),
        "memory_used_percent": vm.percent,
        "load_average": [round(x, 2) for x in psutil.getloadavg()],
        "disk_free_gb": round(disk.free / 2**30, 1),
    }
    ok = database["ok"] and (settings.job_backend != "celery" or queue.get("redis_ok"))
    return SystemStatus(
        status="ok" if ok else "degraded",
        version=__version__,
        environment=settings.environment,
        database=database,
        queue=queue,
        jobs=jobs,
        resources=resources,
        limits={
            "max_concurrent_jobs": settings.max_concurrent_jobs,
            "max_workers_per_job": settings.max_workers_per_job,
            "max_job_seconds": settings.max_job_seconds,
            "max_population": settings.max_population,
            "max_generations": settings.max_generations,
            "max_optuna_trials": settings.max_optuna_trials,
            "max_upload_mb": settings.max_upload_mb,
        },
        storage={"data_dir": str(settings.data_dir), "artifact_dir": str(settings.artifact_dir)},
        dependencies=dependency_versions(),
        code_version=code_version(),
        write_protected=bool(settings.api_key),
        disclaimer=DISCLAIMER,
    )
