"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aether_api import __version__
from aether_api.core.config import get_settings
from aether_api.core.errors import install_error_handlers
from aether_api.core.logging import configure_logging, get_logger
from aether_api.routes import agents, datasets, experiments, jobs, system, tournaments
from aether_api.services import datasets as dataset_jobs  # noqa: F401
from aether_api.services import runner  # noqa: F401  (registers job handlers)
from aether_api.services.jobs import recover_interrupted_jobs, reset_backend
from aether_api.services.storage import ensure_dirs

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.environment == "production" and not settings.api_key:
        raise RuntimeError(
            "AETHER_API_KEY must be set in production (write endpoints trigger CPU-heavy training)"
        )
    ensure_dirs()
    if settings.job_backend == "inline":
        recovered = recover_interrupted_jobs()
        if recovered:
            log.warning("jobs.recovered", count=recovered)
    yield
    reset_backend()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    app = FastAPI(
        title="AETHER API",
        version=__version__,
        description="Autonomous Evolutionary Trading Arena - research API. Simulation only; "
        "no brokerage connectivity.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    install_error_handlers(app)
    v1 = APIRouter(prefix="/api/v1")
    for module in (system, datasets, agents, experiments, tournaments, jobs):
        v1.include_router(module.router)
    app.include_router(v1)
    return app


app = create_app()
