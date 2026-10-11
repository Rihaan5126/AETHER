"""Self-contained experiment export (``aether-experiment-export`` format, version 1).

One JSON document with everything needed to audit or reproduce an experiment:
the immutable manifest (config, dataset hash, code revision, dependency
versions, platform, seed), dataset provenance, the execution and timing
assumptions, the feature set, every agent with its genome, seed and artifact
hash, the lineage, every tournament result (metrics, score components,
warnings), training runs and final-test runs. Equity curves and trade ledgers
are referenced by tournament id rather than inlined (they are available from
the API and are large).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session
from trading_engine import timing
from trading_engine.experiments.runner import FEATURE_CACHE_VERSION
from trading_engine.features import feature_names

from aether_api.database.models import (
    Agent,
    Experiment,
    FinalTestRun,
    ModelLineage,
    Tournament,
    TournamentResult,
    TrainingRun,
)

EXPORT_FORMAT = "aether-experiment-export"
EXPORT_VERSION = 1


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def export_experiment(db: Session, exp: Experiment) -> dict[str, Any]:
    ds = exp.dataset
    evaluation = (exp.config or {}).get("evaluation", {})
    agents = db.scalars(
        select(Agent).where(Agent.experiment_id == exp.id).order_by(Agent.generation, Agent.id)
    ).all()
    lineage = db.scalars(
        select(ModelLineage)
        .where(ModelLineage.experiment_id == exp.id)
        .order_by(ModelLineage.generation, ModelLineage.id)
    ).all()
    tournaments = db.scalars(
        select(Tournament).where(Tournament.experiment_id == exp.id).order_by(Tournament.generation)
    ).all()
    results_by_t: dict[str, list[TournamentResult]] = {t.id: [] for t in tournaments}
    for r in db.scalars(
        select(TournamentResult)
        .where(TournamentResult.tournament_id.in_(list(results_by_t)))
        .order_by(TournamentResult.rank)
    ):
        results_by_t[r.tournament_id].append(r)
    runs = db.scalars(
        select(TrainingRun).where(TrainingRun.experiment_id == exp.id).order_by(TrainingRun.id)
    ).all()
    finals = db.scalars(
        select(FinalTestRun)
        .where(FinalTestRun.experiment_id == exp.id)
        .order_by(FinalTestRun.run_number)
    ).all()
    return {
        "format": EXPORT_FORMAT,
        "format_version": EXPORT_VERSION,
        "exported_at": datetime.now(UTC).isoformat(),
        "experiment": {
            "id": exp.id,
            "name": exp.name,
            "kind": exp.kind,
            "status": exp.status,
            "seed": exp.seed,
            "code_version": exp.code_version,
            "created_at": _iso(exp.created_at),
            "started_at": _iso(exp.started_at),
            "completed_at": _iso(exp.completed_at),
            "wall_seconds": exp.wall_seconds,
            "resource_usage": exp.resource_usage,
            "development_period": exp.development_period,
            "test_period": exp.test_period,
            "test_contaminated": exp.test_contaminated,
        },
        "manifest": exp.manifest,
        "config": exp.config,
        "dataset": {
            "id": ds.id,
            "name": ds.name,
            "source": ds.source,
            "version": ds.version,
            "hash": ds.hash,
            "symbols": ds.symbols,
            "start_date": ds.start_date.isoformat(),
            "end_date": ds.end_date.isoformat(),
            "rows": ds.rows,
            "params": ds.params,
            "retrieved_at": _iso(ds.retrieved_at),
            "validation_issue_counts": _issue_counts(ds.validation),
        },
        "execution_assumptions": {
            "feature_availability": timing.FEATURE_AVAILABILITY,
            "execution_delay_bars": timing.EXECUTION_DELAY_BARS,
            "execution_price": timing.EXECUTION_PRICE_FIELD,
            "trading_days_per_year": timing.TRADING_DAYS_PER_YEAR,
            "label": "open[t+1+h] / open[t+1] - 1 > 0 (h = walk_forward.horizon)",
            "execution": evaluation.get("execution"),
            "risk": evaluation.get("risk"),
            "walk_forward": evaluation.get("walk_forward"),
            "ranking": evaluation.get("ranking"),
            "warning_thresholds": evaluation.get("warnings"),
        },
        "features": {"cache_version": FEATURE_CACHE_VERSION, "names": feature_names()},
        "agents": [
            {
                "id": a.id,
                "name": a.name,
                "algorithm": a.algorithm,
                "generation": a.generation,
                "origin": a.origin,
                "parent_id": a.parent_id,
                "seed": a.seed,
                "is_benchmark": a.is_benchmark,
                "genome": a.genome,
                "artifact": (
                    {
                        "path": a.artifact_path,
                        "sha256": a.artifact_sha256,
                        "files": a.artifact_files,
                    }
                    if a.artifact_path
                    else None
                ),
            }
            for a in agents
        ],
        "lineage": [
            {
                "child_id": rec.child_id,
                "parent_id": rec.parent_id,
                "generation": rec.generation,
                "operation": rec.operation,
                "mutations": rec.mutation_description,
                "genome_before": rec.genome_before,
                "genome_after": rec.genome_after,
                "training_seed": rec.training_seed,
                "validation_score": rec.validation_score,
            }
            for rec in lineage
        ],
        "tournaments": [
            {
                "id": t.id,
                "generation": t.generation,
                "period": t.period,
                "status": t.status,
                "config_fingerprint": t.config_fingerprint,
                "folds": t.folds,
                "benchmark_metrics": t.benchmark_metrics,
                "results": [
                    {
                        "agent_id": r.agent_id,
                        "rank": r.rank,
                        "status": r.status,
                        "status_reason": r.status_reason,
                        "score": r.score,
                        "score_components": r.score_components,
                        "metrics": r.metrics_json,
                        "fold_metrics": r.fold_metrics,
                        "warnings": r.warnings,
                        "evaluation_seconds": r.evaluation_seconds,
                    }
                    for r in results_by_t[t.id]
                ],
            }
            for t in tournaments
        ],
        "training_runs": [
            {
                "agent_id": r.agent_id,
                "tournament_id": r.tournament_id,
                "status": r.status,
                "started_at": _iso(r.started_at),
                "completed_at": _iso(r.completed_at),
                "training_metrics": r.training_metrics,
                "error_message": r.error_message,
            }
            for r in runs
        ],
        "final_test_runs": [
            {
                "run_number": f.run_number,
                "status": f.status,
                "agent_ids": f.agent_ids,
                "test_period": f.test_period,
                "acknowledged_contamination": f.acknowledged_contamination,
                "benchmark_metrics": f.benchmark_metrics,
                "results": f.results,
                "created_at": _iso(f.created_at),
            }
            for f in finals
        ],
    }


def _issue_counts(validation: dict[str, Any] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in (validation or {}).get("issues", []):
        counts[issue.get("severity", "unknown")] = (
            counts.get(issue.get("severity", "unknown"), 0) + 1
        )
    return counts
