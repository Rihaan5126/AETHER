"""Command-line helpers: database setup, data import and reproducible experiments.

python -m aether_api.cli init-db
python -m aether_api.cli demo [--generations 3 --population 8]
python -m aether_api.cli import-yahoo --symbols SPY [--start 2005-01-01 --end 2026-10-09]
python -m aether_api.cli datasets
python -m aether_api.cli experiment --dataset <id> --symbol SPY [--generations 3 ...]
python -m aether_api.cli final-test --experiment <id>
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

from sqlalchemy import select

from aether_api.core.config import get_settings
from aether_api.database.models import Base, Dataset, Experiment
from aether_api.database.session import get_engine, session_scope
from aether_api.schemas.datasets import DatasetImportRequest
from aether_api.schemas.experiments import ExperimentCreateRequest, FinalTestRequest
from aether_api.services import runner  # noqa: F401
from aether_api.services.datasets import import_synthetic, import_yahoo, register_manifest
from aether_api.services.experiments import (
    create_experiment,
    request_final_test,
    start_experiment,
)
from aether_api.services.jobs import run_job
from aether_api.services.storage import ensure_dirs


def init_db() -> None:
    """Create tables directly (development). Production uses `alembic upgrade head`."""
    Base.metadata.create_all(get_engine())


def run_experiment(req: ExperimentCreateRequest, idempotency_key: str) -> dict[str, object]:
    """Create (or reuse, by idempotency key) an experiment and run it in this process."""
    with session_scope() as db:
        exp, _ = create_experiment(db, req, get_settings(), idempotency_key)
        if exp.status == "draft":
            job, _ = start_experiment(db, exp)
            job_id = job.id
        else:
            job_id = None
        exp_id = exp.id
    if job_id:
        run_job(job_id)  # synchronous: runs in this process
    with session_scope() as db:
        final = db.get(Experiment, exp_id)
        assert final is not None
        return {
            "experiment_id": final.id,
            "status": final.status,
            "dataset_id": final.dataset_id,
            "wall_seconds": final.wall_seconds,
        }


def demo(generations: int, population: int, seed: int) -> dict[str, object]:
    ensure_dirs()
    init_db()
    manifest = import_synthetic(
        DatasetImportRequest(
            name="Synthetic demo market",
            source="synthetic",
            symbols=["SYN"],
            start_date=date(2012, 1, 1),
            end_date=date(2024, 12, 31),
        )
    )
    with session_scope() as db:
        ds_id = register_manifest(db, manifest).id
    req = ExperimentCreateRequest(
        name=f"Demo evolution (seed {seed})",
        dataset_id=ds_id,
        symbol="SYN",
        population_size=population,
        generations=generations,
        seed=seed,
    )
    key = f"demo-{manifest.hash[:12]}-{seed}-{generations}-{population}"
    return run_experiment(req, key)


def import_yahoo_dataset(
    symbols: list[str], start: date, end: date, name: str | None
) -> dict[str, object]:
    """Download real daily bars and store them as an immutable, hashed dataset version.

    Re-importing later produces a *new* version if Yahoo's history changed (revisions,
    new bars); existing experiments keep pointing at the exact bytes they used.
    """
    ensure_dirs()
    init_db()
    request = DatasetImportRequest(
        name=name or f"Yahoo {' '.join(s.upper() for s in symbols)} {start}..{end}",
        source="yahoo",
        symbols=symbols,
        start_date=start,
        end_date=end,
    )
    manifest = import_yahoo(request)
    with session_scope() as db:
        ds = register_manifest(db, manifest)
        issues = manifest.validation["issues"]
        return {
            "dataset_id": ds.id,
            "hash": manifest.hash[:12],
            "symbols": ds.symbols,
            "start_date": str(ds.start_date),
            "end_date": str(ds.end_date),
            "rows": ds.rows,
            "validation_warnings": [i["message"] for i in issues if i["severity"] == "warning"],
        }


def final_test(experiment_id: str) -> dict[str, object]:
    """Run the locked final test once, on the default (top-ranked) final-generation agents.

    There is deliberately no override here: a second run must go through the API
    with acknowledge_contamination=true, which marks the experiment contaminated.
    """
    with session_scope() as db:
        exp = db.get(Experiment, experiment_id)
        if exp is None:
            raise SystemExit(f"unknown experiment {experiment_id}")
        run, job = request_final_test(db, exp, FinalTestRequest())
        run_id, job_id = run.id, job.id
    run_job(job_id)
    with session_scope() as db:
        from aether_api.database.models import FinalTestRun  # noqa: PLC0415

        done = db.get(FinalTestRun, run_id)
        assert done is not None
        return {
            "final_test_run": done.id,
            "status": done.status,
            "test_period": done.test_period,
            "benchmark": {
                k: (done.benchmark_metrics or {}).get(k) for k in ("total_return", "sharpe_ratio")
            },
            "agents": [
                {
                    "agent_id": r["agent_id"],
                    "total_return": r["metrics"].get("total_return"),
                    "sharpe_ratio": r["metrics"].get("sharpe_ratio"),
                    "excess_return": r["metrics"].get("excess_return"),
                }
                for r in done.results or []
            ],
        }


def list_datasets() -> list[dict[str, object]]:
    init_db()
    with session_scope() as db:
        rows = db.scalars(select(Dataset).order_by(Dataset.created_at.desc())).all()
        return [
            {
                "id": d.id,
                "name": d.name,
                "source": d.source,
                "symbols": d.symbols,
                "range": f"{d.start_date}..{d.end_date}",
            }
            for d in rows
        ]


def main() -> None:
    parser = argparse.ArgumentParser(prog="aether")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    d = sub.add_parser("demo", help="synthetic-data demo experiment (offline)")
    d.add_argument("--generations", type=int, default=3)
    d.add_argument("--population", type=int, default=8)
    d.add_argument("--seed", type=int, default=42)

    y = sub.add_parser("import-yahoo", help="download real daily bars from Yahoo Finance")
    y.add_argument("--symbols", nargs="+", required=True, help="e.g. SPY QQQ")
    y.add_argument("--start", type=date.fromisoformat, default=date(2005, 1, 1))
    y.add_argument(
        "--end",
        type=date.fromisoformat,
        default=date.today() - timedelta(days=1),
        help="inclusive; default yesterday (today's bar may be incomplete)",
    )
    y.add_argument("--name", default=None)

    sub.add_parser("datasets", help="list stored dataset versions")

    e = sub.add_parser("experiment", help="run an evolution experiment on a stored dataset")
    e.add_argument("--dataset", required=True, help="dataset id (see `aether datasets`)")
    e.add_argument("--symbol", required=True)
    e.add_argument("--name", default=None)
    e.add_argument("--generations", type=int, default=3)
    e.add_argument("--population", type=int, default=8)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--fee-bps", type=float, default=1.0)
    e.add_argument("--slippage-bps", type=float, default=5.0)
    e.add_argument("--folds", type=int, default=3)
    e.add_argument("--test-fraction", type=float, default=0.2)

    ft = sub.add_parser("final-test", help="run the locked final test once")
    ft.add_argument("--experiment", required=True)

    args = parser.parse_args()
    if args.cmd == "init-db":
        init_db()
        print("database initialised")
    elif args.cmd == "demo":
        print(json.dumps(demo(args.generations, args.population, args.seed), indent=2))
    elif args.cmd == "import-yahoo":
        result = import_yahoo_dataset(args.symbols, args.start, args.end, args.name)
        print(json.dumps(result, indent=2))
    elif args.cmd == "datasets":
        print(json.dumps(list_datasets(), indent=2))
    elif args.cmd == "final-test":
        print(json.dumps(final_test(args.experiment), indent=2, default=str))
    else:
        ensure_dirs()
        init_db()
        req = ExperimentCreateRequest(
            name=args.name or f"{args.symbol.upper()} evolution (seed {args.seed})",
            dataset_id=args.dataset,
            symbol=args.symbol.upper(),
            population_size=args.population,
            generations=args.generations,
            seed=args.seed,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            n_folds=args.folds,
            test_fraction=args.test_fraction,
        )
        key = (
            f"cli-{args.dataset}-{req.symbol}-{args.seed}-{args.generations}-"
            f"{args.population}-{args.fee_bps}-{args.slippage_bps}-{args.folds}-"
            f"{args.test_fraction}"
        )
        print(json.dumps(run_experiment(req, key), indent=2))


if __name__ == "__main__":
    main()
