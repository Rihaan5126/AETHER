"""Export stored results as a static, read-only snapshot for the dashboard.

The dashboard's snapshot mode (NEXT_PUBLIC_AETHER_SNAPSHOT=true) reads these files
instead of calling the API, so the results can be hosted on a static/frontend-only
platform such as Vercel with no Python backend, database or training worker.

Every file is a verbatim API response produced by the real FastAPI app (via its
test client, against the configured database). Nothing is recomputed or edited,
except /system/status, which is replaced by a description of the snapshot because
the live response describes the exporting machine (local paths, memory, disk).

Usage (from the repo root):
    uv run python scripts/export_snapshot.py                      # all experiments
    uv run python scripts/export_snapshot.py --experiment exp_... --experiment exp_...
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aether_api.core.config import REPO_ROOT, get_settings
from aether_api.main import app
from fastapi.testclient import TestClient

DEFAULT_OUT = REPO_ROOT / "apps" / "web" / "public" / "snapshot"
PAGE_LIMIT = 200  # API maximum
# max_points values the dashboard requests (see apps/web/lib/queries.ts and features/).
TOURNAMENT_CURVE_POINTS = (600, 1500)
FINAL_TEST_CURVE_POINTS = (800,)
PRICE_POINTS = (600,)


class Exporter:
    def __init__(self, client: TestClient, out: Path) -> None:
        self.client = client
        self.out = out
        self.files = 0
        self.bytes = 0

    def get(self, path: str, **query: Any) -> Any:
        r = self.client.get(f"/api/v1{path}", params=query)
        if r.status_code != 200:
            raise RuntimeError(f"GET {path} {query} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    def get_all(self, path: str, **query: Any) -> dict[str, Any]:
        """Fetch every page of a list endpoint into one page-shaped object."""
        items: list[Any] = []
        offset = 0
        while True:
            page = self.get(path, limit=PAGE_LIMIT, offset=offset, **query)
            items.extend(page["items"])
            offset += PAGE_LIMIT
            if offset >= page["total"]:
                return {"items": items, "total": len(items), "limit": len(items), "offset": 0}

    def write(self, rel: str, data: Any) -> None:
        path = self.out / f"{rel}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        path.write_text(text, encoding="utf-8")
        self.files += 1
        self.bytes += len(text.encode())


def snapshot_status(live: dict[str, Any], generated_at: str, experiments: int) -> dict[str, Any]:
    """System page content for the snapshot: what it is, not the exporting machine."""
    return {
        "status": "ok",
        "version": live["version"],
        "environment": "snapshot",
        "database": {"ok": True, "dialect": "static snapshot", "generated_at": generated_at},
        "queue": {"backend": "none (read-only snapshot)"},
        "jobs": {"queued": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0},
        "resources": {"note": "No training runs here. Results were computed locally."},
        "limits": live["limits"],
        "storage": {"experiments_exported": experiments},
        "dependencies": live["dependencies"],
        "code_version": live["code_version"],
        "write_protected": True,
        "disclaimer": live["disclaimer"],
    }


def check_no_local_paths(out: Path) -> None:
    """Refuse to publish a snapshot that leaks paths from the exporting machine."""
    needles = {str(REPO_ROOT), str(Path.home())}
    needles |= {n.replace("\\", "\\\\") for n in needles}  # JSON-escaped Windows paths
    pattern = re.compile("|".join(re.escape(n) for n in needles), re.IGNORECASE)
    for f in out.rglob("*.json"):
        match = pattern.search(f.read_text(encoding="utf-8"))
        if match:
            raise SystemExit(f"refusing to publish: {f} contains local path {match.group(0)!r}")


def export(out: Path, experiment_ids: list[str] | None) -> dict[str, Any]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")

    with TestClient(app) as client:
        ex = Exporter(client, out)

        experiments = ex.get_all("/experiments")
        if experiment_ids:
            missing = set(experiment_ids) - {e["id"] for e in experiments["items"]}
            if missing:
                raise SystemExit(f"unknown experiment ids: {sorted(missing)}")
            experiments["items"] = [e for e in experiments["items"] if e["id"] in experiment_ids]
        experiments["items"] = [e for e in experiments["items"] if e["status"] == "completed"]
        experiments["total"] = experiments["limit"] = len(experiments["items"])
        keep = {e["id"] for e in experiments["items"]}
        if not keep:
            raise SystemExit("no completed experiments to export")
        ex.write("experiments", experiments)

        dataset_ids: set[str] = set()
        for exp in experiments["items"]:
            eid = exp["id"]
            dataset_ids.add(exp["dataset_id"])
            detail = ex.get(f"/experiments/{eid}")
            ex.write(f"experiments/{eid}", detail)
            ex.write(f"experiments/{eid}/lineage", ex.get(f"/experiments/{eid}/lineage"))
            ex.write(f"experiments/{eid}/analytics", ex.get(f"/experiments/{eid}/analytics"))
            ex.write(f"experiments/{eid}/manifest", ex.get(f"/experiments/{eid}/manifest"))
            for job in detail["jobs"]:
                ex.write(f"jobs/{job['id']}", ex.get(f"/jobs/{job['id']}"))
            for run in detail["final_test_runs"]:
                if run["status"] != "completed":
                    continue
                base = f"/experiments/{eid}/final-test/{run['run_number']}/equity-curves"
                for mp in FINAL_TEST_CURVE_POINTS:
                    ex.write(f"{base[1:]}.mp{mp}", ex.get(base, max_points=mp))

        agents = ex.get_all("/agents")
        agents["items"] = [a for a in agents["items"] if a["experiment_id"] in keep]
        agents["total"] = agents["limit"] = len(agents["items"])
        ex.write("agents", agents)
        for agent in agents["items"]:
            aid = agent["id"]
            ex.write(f"agents/{aid}", ex.get(f"/agents/{aid}"))
            ex.write(f"agents/{aid}/metrics", ex.get(f"/agents/{aid}/metrics"))
            ex.write(f"agents/{aid}/lineage", ex.get(f"/agents/{aid}/lineage"))

        tournaments = ex.get_all("/tournaments")
        tournaments["items"] = [t for t in tournaments["items"] if t["experiment_id"] in keep]
        tournaments["total"] = tournaments["limit"] = len(tournaments["items"])
        ex.write("tournaments", tournaments)
        for t in tournaments["items"]:
            tid = t["id"]
            ex.write(f"tournaments/{tid}", ex.get(f"/tournaments/{tid}"))
            ex.write(f"tournaments/{tid}/leaderboard", ex.get(f"/tournaments/{tid}/leaderboard"))
            for mp in TOURNAMENT_CURVE_POINTS:
                curves = ex.get(f"/tournaments/{tid}/equity-curves", max_points=mp)
                ex.write(f"tournaments/{tid}/equity-curves.mp{mp}", curves)
            ex.write(f"tournaments/{tid}/trades", ex.get_all(f"/tournaments/{tid}/trades"))

        datasets = ex.get_all("/datasets")
        datasets["items"] = [d for d in datasets["items"] if d["id"] in dataset_ids]
        datasets["total"] = datasets["limit"] = len(datasets["items"])
        ex.write("datasets", datasets)
        for ds in datasets["items"]:
            did = ds["id"]
            ex.write(f"datasets/{did}", ex.get(f"/datasets/{did}"))
            for symbol in ds["symbols"]:
                for mp in PRICE_POINTS:
                    prices = ex.get(f"/datasets/{did}/prices", symbol=symbol, max_points=mp)
                    ex.write(f"datasets/{did}/prices.{symbol}.mp{mp}", prices)

        live_status = ex.get("/system/status")
        ex.write("system/status", snapshot_status(live_status, generated_at, len(keep)))

        meta = {
            "generated_at": generated_at,
            "code_version": live_status["code_version"],
            "experiments": [{"id": e["id"], "name": e["name"]} for e in experiments["items"]],
        }
        ex.write("meta", meta)

    check_no_local_paths(out)
    return {"out": str(out), "files": ex.files, "megabytes": round(ex.bytes / 2**20, 2), **meta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--experiment", action="append", dest="experiments", default=None)
    args = parser.parse_args()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    print(f"exporting from {get_settings().database_url.split('///')[0]}///...", file=sys.stderr)
    print(json.dumps(export(args.out, args.experiments), indent=2))


if __name__ == "__main__":
    main()
