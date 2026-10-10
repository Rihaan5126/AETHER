"""Command-line helpers: database setup and a reproducible offline demo.

    python -m aether_api.cli init-db
    python -m aether_api.cli demo [--generations 3 --population 8]
"""

from __future__ import annotations

import argparse
import json
from datetime import date

from aether_api.database.models import Base
from aether_api.database.session import get_engine, session_scope
from aether_api.schemas.datasets import DatasetImportRequest
from aether_api.schemas.experiments import ExperimentCreateRequest
from aether_api.services import runner  # noqa: F401
from aether_api.services.datasets import import_synthetic, register_manifest
from aether_api.services.experiments import create_experiment, start_experiment
from aether_api.services.jobs import run_job
from aether_api.core.config import get_settings
from aether_api.services.storage import ensure_dirs


def init_db() -> None:
    """Create tables directly (development). Production uses `alembic upgrade head`."""
    Base.metadata.create_all(get_engine())


def demo(generations: int, population: int, seed: int) -> dict[str, object]:
    ensure_dirs()
    init_db()
    manifest = import_synthetic(DatasetImportRequest(
        name="Synthetic demo market", source="synthetic", symbols=["SYN"],
        start_date=date(2012, 1, 1), end_date=date(2024, 12, 31)))
    with session_scope() as db:
        ds = register_manifest(db, manifest)
        req = ExperimentCreateRequest(name=f"Demo evolution (seed {seed})", dataset_id=ds.id,
                                      symbol="SYN", population_size=population,
                                      generations=generations, seed=seed)
        exp, _ = create_experiment(db, req, get_settings(), f"demo-{manifest.hash[:12]}-{seed}-"
                                                            f"{generations}-{population}")
        if exp.status == "draft":
            job, _ = start_experiment(db, exp)
            job_id = job.id
        else:
            job_id = None
        exp_id = exp.id
    if job_id:
        run_job(job_id)  # synchronous: runs in this process
    with session_scope() as db:
        from aether_api.database.models import Experiment  # noqa: PLC0415

        exp = db.get(Experiment, exp_id)
        assert exp is not None
        return {"experiment_id": exp.id, "status": exp.status, "dataset": manifest.hash[:12],
                "wall_seconds": exp.wall_seconds}


def main() -> None:
    parser = argparse.ArgumentParser(prog="aether")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    d = sub.add_parser("demo")
    d.add_argument("--generations", type=int, default=3)
    d.add_argument("--population", type=int, default=8)
    d.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.cmd == "init-db":
        init_db()
        print("database initialised")
    else:
        print(json.dumps(demo(args.generations, args.population, args.seed), indent=2))


if __name__ == "__main__":
    main()
