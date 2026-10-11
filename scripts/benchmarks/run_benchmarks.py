"""Reproducible performance benchmarks for AETHER (writes docs/benchmarks/results.json).

Every number is measured on the machine that runs this script; nothing is
estimated. Timings are medians of repeated runs unless noted. Run on an idle
machine:

    uv run python scripts/benchmarks/run_benchmarks.py                # everything
    uv run python scripts/benchmarks/run_benchmarks.py --only sim     # simulator micro-benchmark

``--only sim`` is what the before/after optimisation comparison uses (run once
against the current code and once with PYTHONPATH pointing at an older checkout).
"""

# Imports live inside each section so a partial run loads only what it measures.
# ruff: noqa: PLC0415
from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import statistics
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import psutil

REPO = Path(__file__).resolve().parents[2]
DATASET_ID = "ds_aa0bb65d9164"  # SPY QQQ AAPL MSFT NVDA, Yahoo daily 2005-2026


def timed(fn: Callable[[], Any], repeat: int) -> dict[str, float]:
    samples = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    return {
        "median_s": round(statistics.median(samples), 4),
        "min_s": round(min(samples), 4),
        "max_s": round(max(samples), 4),
        "repeat": repeat,
    }


class PeakRSS:
    """Samples resident memory of this process plus its children (process pools)."""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval, self.peak, self._stop = interval, 0, threading.Event()
        self._proc = psutil.Process()

    def _total(self) -> int:
        total = self._proc.memory_info().rss
        for child in self._proc.children(recursive=True):
            with contextlib.suppress(psutil.Error):
                total += child.memory_info().rss
        return total

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._total())
            self._stop.wait(self.interval)

    def __enter__(self) -> PeakRSS:
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._t.join()

    @property
    def peak_mb(self) -> float:
        return round(self.peak / 2**20, 1)


def load_market(symbol: str = "SPY") -> tuple[Any, Any, str]:
    from aether_api.database.models import Dataset
    from aether_api.database.session import session_scope
    from aether_api.services.storage import dataset_store
    from sqlalchemy import select
    from trading_engine.experiments import ExperimentConfig, prepare_market_data

    with session_scope() as db:
        h = db.scalars(select(Dataset.hash).where(Dataset.id == DATASET_ID)).one()
    dev, test = prepare_market_data(
        dataset_store(), ExperimentConfig(name="bench", dataset_hash=h, symbol=symbol)
    )
    return dev, test, h


# ---- sections ----------------------------------------------------------------------------
def bench_sim(repeat: int) -> dict[str, Any]:
    """Simulator + metrics throughput on the full SPY history (old and new code alike)."""
    import trading_engine
    from trading_engine.agents import Observations, create_agent
    from trading_engine.agents.genomes import MomentumGenome, RandomGenome
    from trading_engine.evaluation import compute_metrics
    from trading_engine.research import random_null_sharpes
    from trading_engine.simulation import simulate
    from trading_engine.training import MarketFrame

    dev, test, _ = load_market("SPY")
    full = MarketFrame(
        "SPY", np_concat(dev.bars, test.bars), np_concat(dev.features, test.features)
    )
    obs = Observations("SPY", full.features, full.bars)
    out: dict[str, Any] = {
        "engine_path": str(Path(trading_engine.__file__).parent),
        "bars": len(full.bars),
    }
    for name, genome, seed in [
        ("random (trade-heavy)", RandomGenome(), 1),
        ("momentum ma_cross (few trades)", MomentumGenome(), 1),
    ]:
        w = create_agent("b", genome, seed).predict(obs)
        res = simulate(full.bars, w, realised_vol=full.features["vol_20"])
        sim = timed(
            lambda w=w: simulate(full.bars, w, realised_vol=full.features["vol_20"]), repeat
        )
        met = timed(lambda res=res: compute_metrics(res), repeat)
        out[name] = {
            "trades": len(res.trades),
            "simulate": sim,
            "bars_per_second": round(len(full.bars) / sim["median_s"]),
            "compute_metrics": met,
        }
    out["random_null_100_on_test_window"] = timed(
        lambda: random_null_sharpes(test.bars, test.features, n=100, seed=1), max(1, repeat // 2)
    )
    out["random_null_100_on_test_window"]["bars"] = len(test.bars)
    return out


def np_concat(a: Any, b: Any) -> Any:
    import pandas as pd

    return pd.concat([a, b])


def bench_data(repeat: int) -> dict[str, Any]:
    from aether_api.services.storage import dataset_store
    from trading_engine.data.schema import symbol_frame
    from trading_engine.experiments.runner import load_features
    from trading_engine.features import compute_features

    _, _, h = load_market("SPY")
    store = dataset_store()
    raw = store.load(h)
    bars = symbol_frame(raw, "SPY")
    return {
        "dataset_rows": len(raw),
        "load_with_sha256_verification": timed(lambda: store.load(h), repeat),
        "load_without_verification": timed(lambda: store.load(h, verify=False), repeat),
        "compute_features_spy": {
            **timed(lambda: compute_features(bars), repeat),
            "bars": len(bars),
        },
        "load_features_cached": timed(lambda: load_features(store, h, "SPY", bars), repeat),
    }


def bench_training(repeat: int) -> dict[str, Any]:
    from trading_engine.agents import (
        FitContext,
        Observations,
        TrainingSet,
        create_agent,
        default_genome,
    )
    from trading_engine.training import WalkForwardConfig, walk_forward_folds
    from trading_engine.training.splits import training_matrix

    dev, _, _ = load_market("SPY")
    fold = walk_forward_folds(dev, WalkForwardConfig())[-1]
    X, y = training_matrix(dev, fold.train_positions, horizon=1, label_cutoff=fold.val_start)
    training = TrainingSet(X, y, dev.bars.iloc[: fold.val_start])
    val = dev.slice_positions(fold.val_start, fold.val_stop)
    obs = Observations("SPY", val.features, dev.bars.iloc[: fold.val_stop])
    out: dict[str, Any] = {"training_rows": len(X), "validation_rows": len(val.bars)}
    for fam in ("random_forest", "xgboost", "mlp", "momentum"):
        agent = create_agent("b", default_genome(fam), 1)
        f = timed(lambda agent=agent: agent.fit(training, FitContext(seed=1)), repeat)
        p = timed(lambda agent=agent: agent.predict(obs), repeat)
        out[fam] = {"fit": f, "predict_validation": p}
    return out


def bench_evaluation(repeat: int) -> dict[str, Any]:
    from trading_engine.agents import default_genome
    from trading_engine.tournaments import AgentSpec, EvaluationConfig
    from trading_engine.tournaments.engine import benchmark_folds, evaluate_agent
    from trading_engine.training import walk_forward_folds

    dev, _, _ = load_market("SPY")
    cfg = EvaluationConfig()
    folds = walk_forward_folds(dev, cfg.walk_forward)
    bench = benchmark_folds(dev, folds, cfg)
    out: dict[str, Any] = {"folds": len(folds)}
    for fam in ("momentum", "random_forest", "xgboost", "mlp"):
        out[fam] = timed(
            lambda fam=fam: evaluate_agent(
                AgentSpec("b", default_genome(fam), 1), dev, folds, cfg, bench
            ),
            max(1, repeat // 2),
        )
    return out


def bench_scaling(populations: list[int], workers: int, serial_up_to: int) -> dict[str, Any]:
    from trading_engine.evolution import EvolutionConfig
    from trading_engine.evolution.engine import initial_population
    from trading_engine.tournaments import AgentSpec, EvaluationConfig, run_tournament

    dev, _, _ = load_market("SPY")
    cfg = EvaluationConfig()
    out: dict[str, Any] = {"workers": workers, "runs": []}
    for pop in populations:
        rng = np.random.default_rng(pop)
        genomes = initial_population(EvolutionConfig(population_size=pop), rng)
        specs = [
            AgentSpec(f"s{i:02d}-{g.family}", g, int(rng.integers(0, 2**31 - 1)))
            for i, g in enumerate(genomes)
        ]
        fams: dict[str, int] = {}
        for g in genomes:
            fams[g.family] = fams.get(g.family, 0) + 1
        for w in ([1] if pop <= serial_up_to else []) + [workers]:
            with PeakRSS() as mem:
                t = time.perf_counter()
                outcome = run_tournament(specs, dev, cfg, max_workers=w)
                wall = time.perf_counter() - t
            out["runs"].append(
                {
                    "population": pop,
                    "max_workers": w,
                    "families": fams,
                    "wall_s": round(wall, 2),
                    "seconds_per_agent": round(wall / pop, 3),
                    "peak_rss_mb_process_tree": mem.peak_mb,
                    "failed": sum(1 for e in outcome.evaluations if e.status == "failed"),
                }
            )
            print(
                f"  scaling pop={pop} workers={w}: {wall:.1f}s, peak {mem.peak_mb} MB", flush=True
            )
    return out


def bench_api(repeat: int) -> dict[str, Any]:
    from aether_api.main import app
    from fastapi.testclient import TestClient

    out: dict[str, Any] = {"note": "in-process TestClient against the local SQLite database"}
    with TestClient(app) as c:
        exps = c.get("/api/v1/experiments?status=completed&limit=1").json()["items"]
        if not exps:
            return {"skipped": "no completed experiment"}
        eid = exps[0]["id"]
        detail = c.get(f"/api/v1/experiments/{eid}").json()
        tid = detail["generations"][-1]["tournament_id"]
        aid = c.get(f"/api/v1/agents?experiment_id={eid}&limit=1").json()["items"][0]["id"]
        endpoints = {
            "health": "/api/v1/health",
            "experiments list": "/api/v1/experiments?limit=50",
            "experiment detail": f"/api/v1/experiments/{eid}",
            "leaderboard": f"/api/v1/tournaments/{tid}/leaderboard",
            "equity curves (600 pts)": f"/api/v1/tournaments/{tid}/equity-curves?max_points=600",
            "trades (50)": f"/api/v1/tournaments/{tid}/trades?limit=50",
            "agents list (200)": "/api/v1/agents?limit=200",
            "agent detail": f"/api/v1/agents/{aid}",
            "analytics (incl. bootstrap)": f"/api/v1/experiments/{eid}/analytics",
            "manifest export": f"/api/v1/experiments/{eid}/manifest",
        }
        for name, url in endpoints.items():
            samples, size = [], 0
            for _ in range(repeat):
                t = time.perf_counter()
                r = c.get(url)
                samples.append((time.perf_counter() - t) * 1000)
                size = len(r.content)
                assert r.status_code == 200, (url, r.status_code)
            samples.sort()
            out[name] = {
                "median_ms": round(statistics.median(samples), 1),
                "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 1),
                "response_kb": round(size / 1024, 1),
                "n": repeat,
            }
        out["experiment_id"] = eid
    return out


def bench_storage() -> dict[str, Any]:
    from aether_api.core.config import get_settings

    s = get_settings()

    def size(p: Path) -> tuple[int, int]:
        files = [f for f in p.rglob("*") if f.is_file()] if p.exists() else []
        return sum(f.stat().st_size for f in files), len(files)

    db_path = (
        Path(s.database_url.split("///", 1)[1]) if s.database_url.startswith("sqlite") else None
    )
    out: dict[str, Any] = {}
    if db_path and db_path.exists():
        out["sqlite_db_mb"] = round(db_path.stat().st_size / 2**20, 2)
    b, n = size(s.data_dir)
    out["datasets_store"] = {"mb": round(b / 2**20, 2), "files": n}
    per_exp = []
    exp_root = s.artifact_dir / "experiments"
    if exp_root.exists():
        for d in sorted(exp_root.iterdir()):
            b, n = size(d)
            per_exp.append({"experiment": d.name, "mb": round(b / 2**20, 2), "files": n})
    out["artifacts_per_experiment"] = per_exp
    fam_sizes: dict[str, list[int]] = {}
    for meta in exp_root.rglob("meta.json") if exp_root.exists() else []:
        fam = json.loads(meta.read_text()).get("family")
        fam_sizes.setdefault(fam, []).append(sum(f.stat().st_size for f in meta.parent.iterdir()))
    out["model_artifact_kb_by_family"] = {
        f: {"n": len(v), "median_kb": round(statistics.median(v) / 1024, 1)}
        for f, v in fam_sizes.items()
    }
    return out


def machine() -> dict[str, Any]:
    from trading_engine.experiments.manifest import code_version, dependency_versions

    freq = psutil.cpu_freq()
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "cpu_max_mhz": round(freq.max) if freq else None,
        "memory_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "python": platform.python_version(),
        "dependencies": dependency_versions(),
        "code_version": code_version(),
        "cpu_load_at_start_percent": psutil.cpu_percent(interval=1.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only", choices=["sim", "data", "training", "evaluation", "scaling", "api", "storage"]
    )
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--populations", default="8,16,32,64")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "benchmarks" / "results.json")
    args = ap.parse_args()
    sections: dict[str, Callable[[], Any]] = {
        "sim": lambda: bench_sim(args.repeat),
        "data": lambda: bench_data(args.repeat),
        "training": lambda: bench_training(max(1, args.repeat // 2)),
        "evaluation": lambda: bench_evaluation(args.repeat),
        "scaling": lambda: bench_scaling(
            [int(p) for p in args.populations.split(",")], args.workers, 16
        ),
        "api": lambda: bench_api(30),
        "storage": bench_storage,
    }
    chosen = [args.only] if args.only else list(sections)
    result: dict[str, Any] = {
        "machine": machine(),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for name in chosen:
        print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)
        result[name] = sections[name]()
    if args.only:
        print(json.dumps(result, indent=2))
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
