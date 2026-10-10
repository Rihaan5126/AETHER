from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="aether-api-tests-"))
os.environ.update(
    {
        "AETHER_ENVIRONMENT": "test",
        "AETHER_DATABASE_URL": f"sqlite:///{_TMP / 'test.db'}",
        "AETHER_DATA_DIR": str(_TMP / "data"),
        "AETHER_ARTIFACT_DIR": str(_TMP / "artifacts"),
        "AETHER_JOB_BACKEND": "inline",
        "AETHER_MAX_CONCURRENT_JOBS": "1",
        "AETHER_WRITE_RATE_LIMIT_PER_MINUTE": "1000",
    }
)

from aether_api.core.config import get_settings  # noqa: E402
from aether_api.core.security import rate_limiter  # noqa: E402
from aether_api.database.models import Base  # noqa: E402
from aether_api.database.session import get_engine  # noqa: E402
from aether_api.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

Base.metadata.create_all(get_engine())


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()
    rate_limiter.reset()


def wait_for_job(client: TestClient, job_id: str, timeout: float = 300) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not finish: {job}")


@pytest.fixture(scope="session")
def dataset_id(client: TestClient) -> str:
    r = client.post(
        "/api/v1/datasets/import",
        json={
            "name": "Synthetic test market",
            "source": "synthetic",
            "symbols": ["SYN", "ALT"],
            "start_date": "2013-01-01",
            "end_date": "2021-12-31",
        },
    )
    assert r.status_code == 202, r.text
    job = wait_for_job(client, r.json()["job_id"])
    assert job["status"] == "completed", job
    return job["result"]["dataset_id"]


SMALL_EXPERIMENT = {
    "symbol": "SYN",
    "families": ["momentum", "xgboost"],
    "population_size": 4,
    "generations": 2,
    "elite_count": 1,
    "immigrant_count": 1,
    "n_folds": 2,
    "seed": 3,
}


def settings_override(**kw: object):  # type: ignore[no-untyped-def]
    base = get_settings()
    return lambda: base.model_copy(update=kw)
