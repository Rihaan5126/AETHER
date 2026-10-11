from __future__ import annotations

import io

from aether_api.core.config import get_settings
from aether_api.main import app
from conftest import SMALL_EXPERIMENT, settings_override, wait_for_job


def test_health_and_status(client):
    assert client.get("/api/v1/health").json()["status"] == "ok"
    status = client.get("/api/v1/system/status").json()
    assert status["database"]["ok"] and status["queue"]["backend"] == "inline"
    assert "simulated" in status["disclaimer"]


def test_openapi_has_no_brokerage_surface(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/v1/experiments/{experiment_id}/start" in paths
    for path in paths:
        assert not any(word in path for word in ("order", "broker", "account", "live"))


def test_structured_errors(client):
    r = client.get("/api/v1/experiments/exp_missing")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = client.post(
        "/api/v1/datasets/import",
        json={
            "name": "x",
            "symbols": ["bad symbol!"],
            "start_date": "2020-01-01",
            "end_date": "2019-01-01",
        },
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_dataset_listing_and_detail(client, dataset_id):
    page = client.get("/api/v1/datasets?limit=5").json()
    assert page["total"] >= 1 and page["limit"] == 5
    detail = client.get(f"/api/v1/datasets/{dataset_id}").json()
    assert detail["symbols"] == ["ALT", "SYN"] and len(detail["hash"]) == 64
    assert detail["validation"]["rows_in"] == detail["rows"]
    prices = client.get(f"/api/v1/datasets/{dataset_id}/prices?symbol=SYN&max_points=50").json()
    assert 0 < len(prices["points"]) <= 60 and prices["downsampled"]


def test_reimporting_identical_data_reuses_version(client, dataset_id):
    r = client.post(
        "/api/v1/datasets/import",
        json={
            "name": "same data again",
            "symbols": ["SYN", "ALT"],
            "start_date": "2013-01-01",
            "end_date": "2021-12-31",
        },
    )
    job = wait_for_job(client, r.json()["job_id"])
    assert job["result"]["dataset_id"] == dataset_id


def test_upload_csv_and_rejections(client):
    import pandas as pd
    from trading_engine.data import SyntheticDataProvider

    raw = SyntheticDataProvider().fetch_ohlcv(
        ["UPL"], pd.Timestamp("2020-01-01"), pd.Timestamp("2021-06-30")
    )
    csv = (
        raw.drop(columns=["symbol", "adjustment", "source"])
        .rename(columns={"timestamp": "Date"})
        .to_csv(index=False)
        .encode()
    )
    r = client.post(
        "/api/v1/datasets/upload",
        data={"name": "uploaded", "symbol": "UPL"},
        files={"file": ("prices.csv", io.BytesIO(csv), "text/csv")},
    )
    assert r.status_code == 201, r.text
    assert r.json()["source"] == "upload" and r.json()["symbols"] == ["UPL"]
    r = client.post(
        "/api/v1/datasets/upload",
        data={"name": "bad"},
        files={"file": ("evil.py", io.BytesIO(b"print(1)"), "text/plain")},
    )
    assert r.status_code == 400
    app.dependency_overrides[get_settings] = settings_override(max_upload_mb=1)
    big = b"x" * (2 * 2**20)
    r = client.post(
        "/api/v1/datasets/upload",
        data={"name": "big", "symbol": "BIG"},
        files={"file": ("big.csv", io.BytesIO(big), "text/csv")},
    )
    assert r.status_code == 413


def test_experiment_validation(client, dataset_id):
    body = {**SMALL_EXPERIMENT, "name": "bad", "dataset_id": dataset_id, "elite_count": 4}
    assert client.post("/api/v1/experiments", json=body).status_code == 422
    body = {**SMALL_EXPERIMENT, "name": "bad", "dataset_id": dataset_id, "symbol": "NOPE"}
    r = client.post("/api/v1/experiments", json=body)
    assert r.status_code == 400 and "available" in r.json()["error"]["details"]
    body = {**SMALL_EXPERIMENT, "name": "bad", "dataset_id": "ds_missing"}
    assert client.post("/api/v1/experiments", json=body).status_code == 404
    app.dependency_overrides[get_settings] = settings_override(max_population=3)
    body = {**SMALL_EXPERIMENT, "name": "too big", "dataset_id": dataset_id}
    assert client.post("/api/v1/experiments", json=body).status_code == 400


def test_experiment_creation_is_idempotent(client, dataset_id):
    body = {**SMALL_EXPERIMENT, "name": "idem", "dataset_id": dataset_id}
    h = {"Idempotency-Key": "key-123"}
    a = client.post("/api/v1/experiments", json=body, headers=h)
    b = client.post("/api/v1/experiments", json=body, headers=h)
    assert a.status_code == 201 and b.status_code == 200 and a.json()["id"] == b.json()["id"]
    c = client.post("/api/v1/experiments", json={**body, "seed": 99}, headers=h)
    assert c.status_code == 409


def test_workload_estimate(client, dataset_id):
    r = client.post(
        "/api/v1/experiments/estimate",
        json={**SMALL_EXPERIMENT, "name": "e", "dataset_id": dataset_id},
    )
    est = r.json()
    assert est["model_fits_upper_bound"] == 4 * 2 * 2 and "no runtime estimate" in est["note"]


def test_write_endpoints_require_api_key_when_configured(client, dataset_id):
    app.dependency_overrides[get_settings] = settings_override(api_key="s3cret")
    body = {**SMALL_EXPERIMENT, "name": "auth", "dataset_id": dataset_id}
    assert client.post("/api/v1/experiments", json=body).status_code == 401
    assert (
        client.post("/api/v1/experiments", json=body, headers={"X-API-Key": "wrong"}).status_code
        == 401
    )
    assert (
        client.post("/api/v1/experiments", json=body, headers={"X-API-Key": "s3cret"}).status_code
        == 201
    )
    assert client.get("/api/v1/experiments").status_code == 200  # reads stay open


def test_write_rate_limit(client, dataset_id):
    app.dependency_overrides[get_settings] = settings_override(write_rate_limit_per_minute=2)
    body = {**SMALL_EXPERIMENT, "name": "rl", "dataset_id": dataset_id}
    codes = [client.post("/api/v1/experiments", json=body).status_code for _ in range(3)]
    assert codes == [201, 201, 429]


def test_interrupted_jobs_are_marked_failed(client):
    from datetime import UTC, datetime, timedelta

    from aether_api.database.models import Job
    from aether_api.database.session import session_scope
    from aether_api.services.jobs import recover_interrupted_jobs

    with session_scope() as db:
        db.add(
            Job(
                id="job_orphan",
                job_type="import_dataset",
                status="running",
                heartbeat_at=datetime.now(UTC) - timedelta(hours=2),
                payload={},
            )
        )
    assert recover_interrupted_jobs() >= 1
    job = client.get("/api/v1/jobs/job_orphan").json()
    assert job["status"] == "failed" and "interrupted" in job["error_message"]


def test_job_queue_is_bounded(client, monkeypatch):
    """Rate limits cap requests per minute; this caps total queued CPU work."""
    from aether_api.core.config import get_settings

    monkeypatch.setattr(get_settings(), "max_pending_jobs", 0)
    r = client.post(
        "/api/v1/datasets/import",
        json={
            "name": "over the limit",
            "source": "synthetic",
            "symbols": ["SYN"],
            "start_date": "2015-01-01",
            "end_date": "2017-12-31",
        },
    )
    assert r.status_code == 429, r.text
    assert r.json()["error"]["code"] == "queue_full"
