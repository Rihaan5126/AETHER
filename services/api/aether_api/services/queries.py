"""Read-side queries: leaderboards, curves, lineage and research analytics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aether_api.core.errors import not_found
from aether_api.database.models import (
    Agent,
    EquityCurve,
    Experiment,
    FinalTestRun,
    ModelLineage,
    OptimizationStudy,
    Tournament,
    TournamentResult,
    TrainingRun,
)
from aether_api.services.runner import BENCHMARK_CURVE_ID
from aether_api.services.storage import read_parquet_artifact
from trading_engine.evolution import numeric_genes
from trading_engine.agents.genomes import GENOME_TYPES

MAX_POINTS = 600


def leaderboard_rows(db: Session, tournament: Tournament) -> list[dict[str, Any]]:
    rows = []
    results = db.execute(
        select(TournamentResult, Agent).join(Agent, Agent.id == TournamentResult.agent_id)
        .where(TournamentResult.tournament_id == tournament.id)
        .order_by(TournamentResult.rank)).all()
    for res, agent in results:
        m = res.metrics_json or {}
        rows.append({
            "rank": res.rank, "agent_id": agent.id, "agent_name": agent.name,
            "algorithm": agent.algorithm, "generation": agent.generation,
            "is_benchmark": agent.is_benchmark, "status": res.status,
            "status_reason": res.status_reason, "score": res.score,
            "score_components": res.score_components or {},
            "total_return": m.get("total_return"), "annualized_return": m.get("annualized_return"),
            "sharpe_ratio": m.get("sharpe_ratio"), "sortino_ratio": m.get("sortino_ratio"),
            "max_drawdown": m.get("max_drawdown"), "turnover": m.get("turnover"),
            "n_trades": m.get("n_trades"), "n_round_trips": m.get("n_round_trips"),
            "win_rate": m.get("win_rate"), "exposure": m.get("exposure"),
            "excess_return": m.get("excess_return"),
            "stability": (res.score_components or {}).get("fold_sharpe_std"),
            "warnings": res.warnings or [],
        })
    return rows


def _downsample(frame: pd.DataFrame, max_points: int) -> tuple[pd.DataFrame, bool]:
    if len(frame) <= max_points:
        return frame, False
    idx = np.unique(np.linspace(0, len(frame) - 1, max_points).round().astype(int))
    return frame.iloc[idx], True


def _curve_file(db: Session, *, tournament_id: str | None = None,
                final_test_run_id: str | None = None) -> pd.DataFrame:
    q = select(EquityCurve)
    q = (q.where(EquityCurve.tournament_id == tournament_id) if tournament_id
         else q.where(EquityCurve.final_test_run_id == final_test_run_id))
    curve = db.scalar(q)
    if curve is None:
        return pd.DataFrame(columns=["agent_id", "timestamp", "equity", "drawdown", "weight",
                                     "fold"])
    return read_parquet_artifact(curve.path, curve.sha256)


def equity_curves(db: Session, *, tournament_id: str | None = None,
                  final_test_run_id: str | None = None, agent_ids: list[str] | None = None,
                  max_points: int = MAX_POINTS) -> dict[str, Any]:
    frame = _curve_file(db, tournament_id=tournament_id, final_test_run_id=final_test_run_id)
    series, bench, downsampled = [], None, False
    for agent_id, sub in frame.groupby("agent_id", sort=False):
        if agent_ids and agent_id not in agent_ids and agent_id != BENCHMARK_CURVE_ID:
            continue
        sub, ds = _downsample(sub.sort_values("timestamp"), max_points)
        downsampled |= ds
        points = [{"t": ts.date().isoformat(), "equity": round(float(e), 2),
                   "drawdown": round(float(d), 5), "weight": round(float(w), 4), "fold": int(f)}
                  for ts, e, d, w, f in zip(sub["timestamp"], sub["equity"], sub["drawdown"],
                                            sub["weight"], sub["fold"], strict=True)]
        entry = {"agent_id": str(agent_id), "points": points}
        if agent_id == BENCHMARK_CURVE_ID:
            bench = entry
        else:
            series.append(entry)
    return {"downsampled": downsampled, "series": series, "benchmark": bench}


def lineage_graph(db: Session, experiment_id: str) -> dict[str, Any]:
    agents = db.scalars(select(Agent).where(Agent.experiment_id == experiment_id)).all()
    latest = _latest_results(db, [a.id for a in agents])
    nodes = [{"agent_id": a.id, "algorithm": a.algorithm, "generation": a.generation,
              "origin": a.origin, "is_benchmark": a.is_benchmark,
              "score": latest.get(a.id, {}).get("score"), "rank": latest.get(a.id, {}).get("rank"),
              "status": latest.get(a.id, {}).get("status")}
             for a in sorted(agents, key=lambda a: (a.generation, a.id))]
    edges = [{"parent_id": e.parent_id, "child_id": e.child_id, "operation": e.operation,
              "mutations": e.mutation_description or []}
             for e in db.scalars(select(ModelLineage).where(
                 ModelLineage.experiment_id == experiment_id,
                 ModelLineage.parent_id.is_not(None)))]
    return {"nodes": nodes, "edges": edges}


def _latest_results(db: Session, agent_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not agent_ids:
        return {}
    out: dict[str, dict[str, Any]] = {}
    rows = db.execute(select(TournamentResult, Tournament.generation)
                      .join(Tournament, Tournament.id == TournamentResult.tournament_id)
                      .where(TournamentResult.agent_id.in_(agent_ids))
                      .order_by(Tournament.generation)).all()
    for res, _gen in rows:
        out[res.agent_id] = {"status": res.status, "rank": res.rank, "score": res.score,
                             "total_return": res.total_return, "sharpe_ratio": res.sharpe_ratio,
                             "max_drawdown": res.max_drawdown, "warnings": res.warnings or []}
    return out


def agent_summary(db: Session, agent: Agent,
                  latest: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    info = (latest or _latest_results(db, [agent.id])).get(agent.id, {})
    return {
        "id": agent.id, "experiment_id": agent.experiment_id, "name": agent.name,
        "algorithm": agent.algorithm, "generation": agent.generation,
        "parent_id": agent.parent_id, "origin": agent.origin, "seed": agent.seed,
        "is_benchmark": agent.is_benchmark, "genome": agent.genome,
        "has_artifact": agent.artifact_path is not None, "created_at": agent.created_at,
        **{k: info.get(k) for k in ("status", "rank", "score", "total_return", "sharpe_ratio",
                                    "max_drawdown")},
        "warnings": info.get("warnings", []),
    }


def list_agent_summaries(db: Session, agents: list[Agent]) -> list[dict[str, Any]]:
    latest = _latest_results(db, [a.id for a in agents])
    return [agent_summary(db, a, latest) for a in agents]


def agent_detail(db: Session, agent_id: str) -> dict[str, Any]:
    agent = db.get(Agent, agent_id)
    if agent is None:
        raise not_found("agent", agent_id)
    runs = db.scalars(select(TrainingRun).where(TrainingRun.agent_id == agent_id)).all()
    lineage = db.scalars(select(ModelLineage).where(ModelLineage.child_id == agent_id)).all()
    children = db.scalars(select(ModelLineage.child_id).where(
        ModelLineage.parent_id == agent_id)).all()
    return {
        **agent_summary(db, agent),
        "artifact_path": agent.artifact_path, "artifact_sha256": agent.artifact_sha256,
        "training_runs": [{"id": r.id, "status": r.status, "tournament_id": r.tournament_id,
                           "metrics": r.training_metrics, "error": r.error_message}
                          for r in runs],
        "lineage": [{"parent_id": rec.parent_id, "operation": rec.operation,
                     "mutations": rec.mutation_description, "genome_before": rec.genome_before,
                     "genome_after": rec.genome_after, "training_seed": rec.training_seed,
                     "validation_score": rec.validation_score} for rec in lineage],
        "children": sorted(set(children)),
    }


def agent_metrics(db: Session, agent_id: str) -> dict[str, Any]:
    if db.get(Agent, agent_id) is None:
        raise not_found("agent", agent_id)
    rows = db.execute(select(TournamentResult, Tournament)
                      .join(Tournament, Tournament.id == TournamentResult.tournament_id)
                      .where(TournamentResult.agent_id == agent_id)).all()
    results = [{"tournament_id": t.id, "generation": t.generation, "period": t.period,
                "rank": r.rank, "score": r.score, "status": r.status,
                "status_reason": r.status_reason, "metrics": r.metrics_json,
                "fold_metrics": r.fold_metrics, "score_components": r.score_components,
                "warnings": r.warnings, "risk_events": (r.risk_events or [])[:50]}
               for r, t in rows]
    final = []
    for run in db.scalars(select(FinalTestRun).where(FinalTestRun.status == "completed")):
        for entry in run.results or []:
            if entry.get("agent_id") == agent_id:
                final.append({"run_number": run.run_number, "test_period": run.test_period,
                              "metrics": entry["metrics"],
                              "benchmark_metrics": run.benchmark_metrics,
                              "acknowledged_contamination": run.acknowledged_contamination})
    return {"agent_id": agent_id, "results": results, "final_test": final}


# ---- research analytics -------------------------------------------------------------------
def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 5 or len(set(x)) < 3:
        return None
    rx, ry = pd.Series(x).rank(), pd.Series(y).rank()
    c = rx.corr(ry)
    return None if c is None or math.isnan(c) else round(float(c), 3)


def _regime_returns(curves: pd.DataFrame, agent_ids: list[str]) -> list[dict[str, Any]]:
    bench = curves[curves["agent_id"] == BENCHMARK_CURVE_ID].set_index("timestamp")["equity"]
    if bench.empty:
        return []
    b_ret = bench.pct_change()
    trend = bench.pct_change(60)
    vol = b_ret.rolling(60).std()
    med_vol = vol.median()
    labels = {
        "uptrend": trend > 0, "downtrend": trend <= 0,
        "high_volatility": vol > med_vol, "low_volatility": vol <= med_vol,
    }
    out = []
    for agent_id in [*agent_ids, BENCHMARK_CURVE_ID]:
        eq = curves[curves["agent_id"] == agent_id].set_index("timestamp")["equity"]
        r = eq.pct_change()
        row: dict[str, Any] = {"agent_id": agent_id}
        for name, mask in labels.items():
            m = mask.reindex(r.index).fillna(False).astype(bool) & trend.reindex(r.index).notna()
            sub = r[m].dropna()
            row[name] = round(float(sub.mean() * 252), 4) if len(sub) >= 20 else None
            row[f"{name}_sessions"] = len(sub)
        out.append(row)
    return out


def experiment_analytics(db: Session, exp: Experiment) -> dict[str, Any]:
    tournaments = db.scalars(select(Tournament).where(Tournament.experiment_id == exp.id)
                             .order_by(Tournament.generation)).all()
    if not tournaments:
        return {"experiment_id": exp.id, "available": False}
    generations, all_scores = [], {}
    for t in tournaments:
        rows = leaderboard_rows(db, t)
        scores = [r["score"] for r in rows if r["score"] is not None and not r["is_benchmark"]]
        rand = next((r for r in rows if r["algorithm"] == "random"), None)
        bh = next((r for r in rows if r["algorithm"] == "buy_and_hold"), None)
        generations.append({
            "generation": t.generation, "tournament_id": t.id,
            "agents": [{"agent_id": r["agent_id"], "algorithm": r["algorithm"],
                        "score": r["score"], "status": r["status"],
                        "sharpe_ratio": r["sharpe_ratio"]} for r in rows],
            "best_score": max(scores) if scores else None,
            "median_score": float(np.median(scores)) if scores else None,
            "random_score": rand["score"] if rand else None,
            "buy_and_hold_score": bh["score"] if bh else None,
        })
        for r in rows:
            all_scores[r["agent_id"]] = r
    final_t = tournaments[-1]
    final_rows = [r for r in leaderboard_rows(db, final_t)]
    results = {r.agent_id: r for r in db.scalars(select(TournamentResult).where(
        TournamentResult.tournament_id == final_t.id))}
    folds = []
    for row in final_rows:
        res = results[row["agent_id"]]
        folds.append({"agent_id": row["agent_id"], "algorithm": row["algorithm"],
                      "is_benchmark": row["is_benchmark"],
                      "folds": [{"fold": f.get("fold"), "sharpe_ratio": f.get("sharpe_ratio"),
                                 "total_return": f.get("total_return"),
                                 "max_drawdown": f.get("max_drawdown"),
                                 "benchmark_return": f.get("benchmark_return"),
                                 "val_start": f.get("val_start"), "val_end": f.get("val_end")}
                                for f in (res.fold_metrics or [])]})

    # Variance by family over every unique evaluated agent (benchmarks included for scale).
    agents = {a.id: a for a in db.scalars(select(Agent).where(Agent.experiment_id == exp.id))}
    by_family: dict[str, list[float]] = {}
    for aid, row in all_scores.items():
        if row["score"] is not None:
            by_family.setdefault(agents[aid].algorithm, []).append(row["score"])
    variance = [{"algorithm": fam, "n": len(v), "mean": float(np.mean(v)),
                 "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                 "min": float(np.min(v)), "max": float(np.max(v))}
                for fam, v in sorted(by_family.items())]

    # Parameter sensitivity (Spearman rank correlation of gene value vs score).
    sensitivity = []
    for fam in sorted({a.algorithm for a in agents.values()} - {"random", "buy_and_hold"}):
        members = [(agents[aid].genome, row["score"]) for aid, row in all_scores.items()
                   if agents[aid].algorithm == fam and row["score"] is not None]
        genes = numeric_genes(GENOME_TYPES[fam])
        for gene in genes:
            xs = [float(g[gene]) for g, _ in members]
            ys = [float(s) for _, s in members]
            sensitivity.append({"algorithm": fam, "gene": gene, "n": len(xs),
                                "spearman": _spearman(xs, ys)})
        if fam != "momentum":
            xs = [float(g["policy"]["threshold"]) for g, _ in members]
            sensitivity.append({"algorithm": fam, "gene": "policy.threshold", "n": len(xs),
                                "spearman": _spearman(xs, [float(s) for _, s in members])})

    # Generalisation gap from completed final-test runs.
    gap = []
    runs = db.scalars(select(FinalTestRun).where(FinalTestRun.experiment_id == exp.id,
                                                 FinalTestRun.status == "completed")
                      .order_by(FinalTestRun.run_number)).all()
    for run in runs:
        for entry in run.results or []:
            aid = entry.get("agent_id")
            if aid not in all_scores:
                continue
            val = all_scores[aid]
            test_m = entry["metrics"]
            gap.append({"run_number": run.run_number, "agent_id": aid,
                        "algorithm": agents[aid].algorithm,
                        "validation_sharpe": val["sharpe_ratio"],
                        "test_sharpe": test_m.get("sharpe_ratio"),
                        "validation_return": val["total_return"],
                        "test_return": test_m.get("total_return"),
                        "validation_excess": val["excess_return"],
                        "test_excess": test_m.get("excess_return"),
                        "contaminated": run.acknowledged_contamination})

    curves = _curve_file(db, tournament_id=final_t.id)
    regime = _regime_returns(curves, [r["agent_id"] for r in final_rows])

    runs_all = db.scalars(select(TrainingRun).where(TrainingRun.experiment_id == exp.id)).all()
    n_unique = sum(1 for r in runs_all if not (r.training_metrics or {}).get("cached_from"))
    val_sessions = int((curves["agent_id"] == BENCHMARK_CURVE_ID).sum())
    years = max(val_sessions / 252, 1e-6)
    n_eval = max(n_unique, 1)
    expected_max = math.sqrt(2 * math.log(n_eval)) / math.sqrt(years) if n_eval > 1 else None
    best_sharpe = max((r["sharpe_ratio"] for r in all_scores.values()
                       if r["sharpe_ratio"] is not None and not r["is_benchmark"]), default=None)
    random_rows = [r for r in all_scores.values() if r["algorithm"] == "random"]
    multiple = {
        "agents_evaluated": n_eval, "unique_training_runs": n_unique,
        "validation_years": round(years, 2),
        "expected_max_sharpe_under_null": round(expected_max, 3) if expected_max else None,
        "best_observed_sharpe": best_sharpe,
        "random_baseline_sharpe": random_rows[-1]["sharpe_ratio"] if random_rows else None,
        "explanation": (
            "If N strategies with no true edge are evaluated on the same data, the best "
            "annualised Sharpe ratio is expected to reach roughly sqrt(2 ln N) / sqrt(years) "
            "by chance alone. Treat any best result below that threshold as indistinguishable "
            "from luck."),
    }
    studies = db.scalars(select(OptimizationStudy).where(
        OptimizationStudy.experiment_id == exp.id)).all()
    return {
        "experiment_id": exp.id, "available": True, "generations": generations,
        "fold_performance": folds, "family_variance": variance,
        "parameter_sensitivity": sensitivity, "generalisation": gap, "regimes": regime,
        "multiple_comparisons": multiple,
        "optimization": [{"id": s.id, "family": s.family, "status": s.status,
                          "best_value": s.best_value, "best_params": s.best_params,
                          "trials": [{"number": t.number, "state": t.state, "value": t.value,
                                      "fold_scores": t.fold_scores,
                                      "duration_seconds": t.duration_seconds}
                                     for t in s.trials]} for s in studies],
    }
