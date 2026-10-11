"""End-to-end lifecycle: synthetic data -> experiment -> training -> tournaments ->
evolution -> persistence -> API retrieval -> locked final test. No network needed."""

from __future__ import annotations

import time

import pytest
from conftest import SMALL_EXPERIMENT, wait_for_job

API = "/api/v1"


@pytest.fixture(scope="module")
def completed_experiment(client, dataset_id):
    body = {**SMALL_EXPERIMENT, "name": "integration", "dataset_id": dataset_id}
    exp = client.post(f"{API}/experiments", json=body).json()
    assert exp["status"] == "draft"
    first = client.post(f"{API}/experiments/{exp['id']}/start")
    assert first.status_code == 202
    second = client.post(f"{API}/experiments/{exp['id']}/start")
    assert second.json()["job_id"] == first.json()["job_id"]  # idempotent start
    job = wait_for_job(client, first.json()["job_id"])
    assert job["status"] == "completed", job
    return client.get(f"{API}/experiments/{exp['id']}").json()


def test_experiment_record(completed_experiment):
    exp = completed_experiment
    assert exp["status"] == "completed" and exp["current_generation"] == 1
    assert len(exp["generations"]) == 2
    m = exp["manifest"]
    assert m["seed"] == 3 and m["dataset_hash"] == exp["config"]["dataset_hash"]
    assert m["dependencies"]["xgboost"] and m["code_version"]
    assert (
        exp["test_period"]["locked"]
        and exp["development_period"]["end"] < exp["test_period"]["start"]
    )
    assert exp["resource_usage"]["max_rss_mb"] > 0


def test_leaderboard_and_results(client, completed_experiment):
    gen = completed_experiment["generations"][-1]
    board = client.get(f"{API}/tournaments/{gen['tournament_id']}/leaderboard").json()
    rows = board["rows"]
    assert board["period"] == "validation" and len(rows) == 4 + 2
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    assert {r["algorithm"] for r in rows if r["is_benchmark"]} == {"random", "buy_and_hold"}
    for r in rows:
        assert r["status"] in {"eligible", "benchmark", "insufficient_evidence", "excluded"}
        assert r["n_trades"] is not None and "mean_fold_sharpe" in r["score_components"]
    curves = client.get(f"{API}/tournaments/{gen['tournament_id']}/equity-curves").json()
    assert len(curves["series"]) == 6 and curves["benchmark"]["points"]
    trades = client.get(f"{API}/tournaments/{gen['tournament_id']}/trades?limit=10").json()
    assert (
        trades["total"] > 0 and trades["items"][0]["timestamp"] > trades["items"][0]["signal_time"]
    )


def test_lineage_and_agents(client, completed_experiment):
    exp_id = completed_experiment["id"]
    graph = client.get(f"{API}/experiments/{exp_id}/lineage").json()
    gens = {n["generation"] for n in graph["nodes"]}
    assert gens == {0, 1} and graph["edges"]
    child = next(e for e in graph["edges"])
    detail = client.get(f"{API}/agents/{child['child_id']}").json()
    assert detail["parent_id"] and detail["lineage"][0]["genome_after"]
    sub = client.get(f"{API}/agents/{child['child_id']}/lineage").json()
    assert child["parent_id"] in {n["agent_id"] for n in sub["nodes"]}
    final_gen = client.get(f"{API}/agents?experiment_id={exp_id}&generation=1").json()
    assert final_gen["total"] == 6 and all(a["has_artifact"] for a in final_gen["items"])
    metrics = client.get(f"{API}/agents/{child['child_id']}/metrics").json()
    assert metrics["results"][0]["fold_metrics"]


def test_final_test_is_locked(client, completed_experiment):
    exp_id = completed_experiment["id"]
    r = client.post(f"{API}/experiments/{exp_id}/final-test", json={})
    assert r.status_code == 202
    job = wait_for_job(client, r.json()["job_id"])
    assert job["status"] == "completed", job
    exp = client.get(f"{API}/experiments/{exp_id}").json()
    run = exp["final_test_runs"][0]
    assert run["status"] == "completed" and run["results"]
    assert run["test_period"]["start"] == exp["test_period"]["start"]
    again = client.post(f"{API}/experiments/{exp_id}/final-test", json={})
    assert again.status_code == 409 and again.json()["error"]["code"] == "final_test_locked"
    ack = client.post(
        f"{API}/experiments/{exp_id}/final-test", json={"acknowledge_contamination": True}
    )
    assert ack.status_code == 202
    wait_for_job(client, ack.json()["job_id"])
    assert client.get(f"{API}/experiments/{exp_id}").json()["test_contaminated"] is True
    curves = client.get(f"{API}/experiments/{exp_id}/final-test/1/equity-curves").json()
    assert curves["period"] == "test" and curves["series"]


def test_analytics(client, completed_experiment):
    a = client.get(f"{API}/experiments/{completed_experiment['id']}/analytics").json()
    assert a["available"] and len(a["generations"]) == 2
    assert a["multiple_comparisons"]["agents_evaluated"] >= 4
    assert a["regimes"] and a["fold_performance"]
    unc = a["uncertainty"]
    assert {u["algorithm"] for u in unc} >= {"buy_and_hold"}
    assert "random" not in {u["algorithm"] for u in unc}
    for u in unc:
        s = u["sharpe"]
        if s["estimate"] is not None:
            assert s["low"] <= s["estimate"] <= s["high"]
            assert 0.0 <= s["prob_positive"] <= 1.0
        if u["algorithm"] != "buy_and_hold":
            assert "train_sharpe" in u


def test_optimisation_study(client, completed_experiment):
    exp_id = completed_experiment["id"]
    r = client.post(
        f"{API}/experiments/{exp_id}/optimize", json={"family": "momentum", "n_trials": 4}
    )
    assert r.status_code == 202
    job = wait_for_job(client, r.json()["job_id"])
    assert job["status"] == "completed", job
    study = client.get(f"{API}/experiments/{exp_id}").json()["optimization_studies"][0]
    assert study["n_trials"] == 4 and study["status"] == "completed"


def test_standalone_tournament(client, dataset_id):
    r = client.post(
        f"{API}/tournaments",
        json={
            "name": "baselines",
            "dataset_id": dataset_id,
            "symbol": "ALT",
            "n_folds": 2,
            "agents": [
                {"family": "buy_and_hold"},
                {"family": "random", "seed": 1},
                {"family": "momentum", "genome": {"fast_window": 10, "slow_window": 50}},
            ],
        },
    )
    assert r.status_code == 202, r.text
    job = wait_for_job(client, r.json()["job_id"])
    assert job["status"] == "completed", job
    t = client.get(f"{API}/tournaments?experiment_id={r.json()['experiment_id']}").json()
    board = client.get(f"{API}/tournaments/{t['items'][0]['id']}/leaderboard").json()
    assert len(board["rows"]) == 3
    bad = client.post(
        f"{API}/tournaments",
        json={
            "name": "bad",
            "dataset_id": dataset_id,
            "symbol": "ALT",
            "agents": [{"family": "momentum", "genome": {"fast_window": 90, "slow_window": 50}}],
        },
    )
    assert bad.status_code == 400


def test_cancellation(client, dataset_id):
    body = {
        **SMALL_EXPERIMENT,
        "name": "to cancel",
        "dataset_id": dataset_id,
        "families": ["mlp", "random_forest"],
        "population_size": 8,
        "generations": 3,
    }
    exp = client.post(f"{API}/experiments", json=body).json()
    job_id = client.post(f"{API}/experiments/{exp['id']}/start").json()["job_id"]
    deadline = time.monotonic() + 60
    while client.get(f"{API}/jobs/{job_id}").json()["status"] == "queued":
        assert time.monotonic() < deadline
        time.sleep(0.1)
    assert client.post(f"{API}/experiments/{exp['id']}/cancel").status_code == 200
    job = wait_for_job(client, job_id)
    assert job["status"] == "cancelled"
    assert client.get(f"{API}/experiments/{exp['id']}").json()["status"] == "cancelled"
    # completed/cancelled experiments are immutable: restarting is refused
    assert client.post(f"{API}/experiments/{exp['id']}/start").status_code == 409


def test_manifest_export_is_self_contained(client, completed_experiment):
    exp = completed_experiment
    r = client.get(f"{API}/experiments/{exp['id']}/manifest")
    assert r.status_code == 200
    assert f'filename="aether-{exp["id"]}.json"' in r.headers["content-disposition"]
    doc = r.json()
    assert doc["format"] == "aether-experiment-export" and doc["format_version"] == 1
    assert (
        doc["dataset"]["hash"] == exp["config"]["dataset_hash"] == doc["manifest"]["dataset_hash"]
    )
    assert doc["manifest"]["code_version"] and doc["manifest"]["dependencies"]["torch"]
    ea = doc["execution_assumptions"]
    assert ea["execution_delay_bars"] == 1 and ea["execution_price"] == "open"
    assert ea["execution"]["fee_bps"] == 1.0 and ea["walk_forward"]["n_folds"] == 2
    assert len(doc["features"]["names"]) == 25
    last_gen = max(a["generation"] for a in doc["agents"])
    final_learned = [
        a for a in doc["agents"] if a["generation"] == last_gen and a["algorithm"] == "xgboost"
    ]
    assert all(len(a["artifact"]["sha256"]) == 64 for a in final_learned)
    assert len(doc["tournaments"]) == 2 and all(len(t["results"]) == 6 for t in doc["tournaments"])
    assert all(rec["genome_after"] for rec in doc["lineage"])


def _comparable(doc: dict) -> list[list[tuple]]:
    """Tournament outcomes without run-specific ids or timings."""
    genomes = {a["id"]: (a["algorithm"], a["genome"], a["seed"]) for a in doc["agents"]}
    return [
        [
            (
                genomes[r["agent_id"]],
                r["rank"],
                r["status"],
                r["score"],
                r["metrics"],
            )
            for r in t["results"]
        ]
        for t in doc["tournaments"]
    ]


def test_identical_config_reproduces_identical_results(client, dataset_id, completed_experiment):
    """Run the same configuration again: every genome, rank, score and metric must match.

    Determinism is exact here (single-threaded XGBoost/RF, deterministic torch), so
    the comparison is exact; see docs/REPRODUCIBILITY.md for tolerances elsewhere.
    """
    body = {**SMALL_EXPERIMENT, "name": "integration rerun", "dataset_id": dataset_id}
    exp = client.post(f"{API}/experiments", json=body).json()
    job = wait_for_job(client, client.post(f"{API}/experiments/{exp['id']}/start").json()["job_id"])
    assert job["status"] == "completed", job
    first = client.get(f"{API}/experiments/{completed_experiment['id']}/manifest").json()
    second = client.get(f"{API}/experiments/{exp['id']}/manifest").json()
    assert first["manifest"]["config"] == {**second["manifest"]["config"], "name": "integration"}
    assert _comparable(first) == _comparable(second)
