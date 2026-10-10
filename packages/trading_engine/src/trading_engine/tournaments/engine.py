"""Tournament engine: identical walk-forward evaluation for every agent.

Every agent in a tournament receives the same development data object, the same
folds, the same :class:`EvaluationConfig` (capital, fees, slippage, risk limits,
execution rule) and is simulated independently. For each fold the agent is
freshly created, fitted on that fold's (purged) training rows and simulated on the
fold's validation window, starting from all cash.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from trading_engine.agents import (
    BaseTradingAgent,
    FitContext,
    Observations,
    TrainingSet,
    create_agent,
    genome_to_dict,
    parse_genome,
)
from trading_engine.agents.genomes import BENCHMARK_FAMILIES, BuyAndHoldGenome
from trading_engine.evaluation import (
    WarningThresholds,
    compute_metrics,
    round_trips,
    statistical_warnings,
)
from trading_engine.risk import RiskConfig
from trading_engine.simulation import ExecutionConfig, SimulationResult, simulate
from trading_engine.tournaments.ranking import RankingPolicy, rank_evaluations
from trading_engine.training.splits import (
    DevelopmentData,
    Fold,
    WalkForwardConfig,
    training_matrix,
    walk_forward_folds,
)


class EvaluationConfig(BaseModel):
    """Everything that must be identical for all agents in a tournament."""

    model_config = ConfigDict(frozen=True)

    execution: ExecutionConfig = ExecutionConfig()
    risk: RiskConfig = RiskConfig()
    walk_forward: WalkForwardConfig = WalkForwardConfig()
    ranking: RankingPolicy = RankingPolicy()
    warnings: WarningThresholds = WarningThresholds()
    max_fit_seconds: float | None = Field(300.0, gt=0)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    genome: BaseModel
    seed: int
    name: str | None = None

    @property
    def family(self) -> str:
        return str(self.genome.family)  # type: ignore[attr-defined]


@dataclass
class AgentEvaluation:
    agent_id: str
    family: str
    genome: dict[str, Any]
    seed: int
    is_benchmark: bool
    fold_metrics: list[dict[str, Any]] = field(default_factory=list)
    fold_training: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[dict[str, str]] = field(default_factory=list)
    equity: pd.DataFrame | None = None
    trades: pd.DataFrame | None = None
    rejected_orders: int = 0
    risk_events: list[dict[str, Any]] = field(default_factory=list)
    score: float | None = None
    score_components: dict[str, float] = field(default_factory=dict)
    status: str = "pending"
    status_reason: str = ""
    rank: int | None = None
    error: str | None = None
    wall_seconds: float = 0.0
    cached_from: str | None = None  # set when an identical (genome, seed) was evaluated before


@dataclass
class TournamentOutcome:
    config_fingerprint: str
    folds: list[dict[str, Any]]
    evaluations: list[AgentEvaluation]
    benchmark_equity: pd.DataFrame


def _observations(dev: DevelopmentData, fold: Fold) -> tuple[Observations, pd.DataFrame]:
    val = dev.slice_positions(fold.val_start, fold.val_stop)
    obs = Observations(dev.symbol, val.features, dev.bars.iloc[: fold.val_stop])
    return obs, val.bars


def _fit_for_fold(agent: BaseTradingAgent, dev: DevelopmentData, fold: Fold,
                  cfg: EvaluationConfig) -> dict[str, Any]:
    X, y = training_matrix(dev, fold.train_positions, horizon=cfg.walk_forward.horizon,
                           label_cutoff=fold.val_start)
    history = dev.bars.iloc[: fold.train_positions[-1] + 1]
    return agent.fit(TrainingSet(X, y, history),
                     FitContext(seed=agent.seed, horizon=cfg.walk_forward.horizon,
                                max_seconds=cfg.max_fit_seconds))


def _simulate_fold(agent: BaseTradingAgent, dev: DevelopmentData, fold: Fold,
                   cfg: EvaluationConfig) -> SimulationResult:
    obs, bars = _observations(dev, fold)
    weights = agent.predict(obs)
    return simulate(bars, weights, execution=cfg.execution, risk=cfg.risk,
                    realised_vol=obs.features["vol_20"])


def benchmark_folds(dev: DevelopmentData, folds: Sequence[Fold],
                    cfg: EvaluationConfig) -> list[SimulationResult]:
    bench = create_agent("benchmark", BuyAndHoldGenome(), 0)
    return [_simulate_fold(bench, dev, f, cfg) for f in folds]


def stitch(results: Sequence[SimulationResult], initial: float) -> SimulationResult:
    """Chain independently simulated folds into one compounded validation curve."""
    frames, trades, rejected, events = [], [], [], []
    level = initial
    for k, res in enumerate(results):
        factor = level / res.config.initial_capital
        eq = res.equity.copy()
        for col in ("cash", "position_value", "equity", "quantity"):
            eq[col] = eq[col] * factor
        eq["fold"] = k
        frames.append(eq)
        for t in res.trades:
            trades.append(t.__class__(**{**t.__dict__,
                                         "quantity": t.quantity * factor,
                                         "notional": t.notional * factor,
                                         "fee": t.fee * factor,
                                         "slippage_cost": t.slippage_cost * factor,
                                         "position_after": t.position_after * factor,
                                         "cash_after": t.cash_after * factor}))
        rejected += res.rejected_orders
        events += res.risk_events
        level = float(eq["equity"].iloc[-1])
    equity = pd.concat(frames)
    equity["drawdown"] = equity["equity"] / equity["equity"].cummax().clip(lower=initial) - 1
    first = results[0]
    return SimulationResult(equity, trades, rejected, events, first.config, first.risk)


class EvaluationAborted(Exception):  # noqa: N818 - control-flow signal, not an error
    """Raised by an ``on_fold`` callback to stop evaluating an agent (e.g. pruning)."""


def evaluate_agent(
    spec: AgentSpec,
    dev: DevelopmentData,
    folds: Sequence[Fold],
    cfg: EvaluationConfig,
    benchmark: Sequence[SimulationResult] | None = None,
    on_fold: Callable[[int, dict[str, Any]], None] | None = None,
) -> AgentEvaluation:
    started = time.perf_counter()
    ev = AgentEvaluation(
        agent_id=spec.agent_id, family=spec.family, genome=genome_to_dict(spec.genome),
        seed=spec.seed, is_benchmark=spec.family in BENCHMARK_FAMILIES,
    )
    bench = list(benchmark) if benchmark is not None else benchmark_folds(dev, folds, cfg)
    try:
        results = []
        for fold, bench_res in zip(folds, bench, strict=True):
            agent = create_agent(spec.agent_id, spec.genome, spec.seed)
            training = _fit_for_fold(agent, dev, fold, cfg)
            res = _simulate_fold(agent, dev, fold, cfg)
            m = compute_metrics(res, bench_res.equity["equity"])
            ev.fold_metrics.append({"fold": fold.number, **fold.to_dict(), **m.to_dict()})
            ev.fold_training.append({"fold": fold.number, **_jsonable(training)})
            results.append(res)
            if on_fold:
                on_fold(fold.number, ev.fold_metrics[-1])
        stitched = stitch(results, cfg.execution.initial_capital)
        bench_stitched = stitch(bench, cfg.execution.initial_capital)
        trips = [t for r in results for t in round_trips(r)]
        overall = compute_metrics(stitched, bench_stitched.equity["equity"], trips=trips)
        ev.metrics = overall.to_dict()
        from trading_engine.evaluation.metrics import PerformanceMetrics  # noqa: PLC0415

        fold_objs = [PerformanceMetrics(**{k: m[k] for k in PerformanceMetrics.__dataclass_fields__})
                     for m in ev.fold_metrics]
        ev.warnings = statistical_warnings(overall, fold_objs,
                                           is_active=spec.family != "buy_and_hold",
                                           thresholds=cfg.warnings)
        ev.equity = stitched.equity
        ev.trades = stitched.trades_frame()
        ev.rejected_orders = len(stitched.rejected_orders)
        ev.risk_events = stitched.risk_events[:500]
        ev.status = "evaluated"
    except EvaluationAborted:
        raise
    except Exception as exc:  # recorded, never swallowed: shows up as an excluded agent
        ev.error = f"{type(exc).__name__}: {exc}"
        ev.status = "failed"
        ev.fold_training.append({"traceback": traceback.format_exc(limit=5)})
    ev.wall_seconds = round(time.perf_counter() - started, 3)
    return ev


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and not np.isfinite(v):
            v = None
        out[k] = v
    return out


def _evaluate_payload(args: tuple[dict[str, Any], DevelopmentData, list[Fold], EvaluationConfig,
                                  list[SimulationResult]]) -> AgentEvaluation:
    payload, dev, folds, cfg, bench = args
    spec = AgentSpec(payload["agent_id"], parse_genome(payload["genome"]), payload["seed"])
    return evaluate_agent(spec, dev, folds, cfg, bench)


def run_tournament(
    specs: Sequence[AgentSpec],
    dev: DevelopmentData,
    cfg: EvaluationConfig,
    *,
    max_workers: int = 1,
    on_progress: Callable[[int, int, AgentEvaluation], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> TournamentOutcome:
    """Evaluate ``specs`` on development data only and rank them."""
    if not isinstance(dev, DevelopmentData):
        raise TypeError("tournaments only accept DevelopmentData (final test is locked)")
    folds = walk_forward_folds(dev, cfg.walk_forward)
    bench = benchmark_folds(dev, folds, cfg)
    evaluations: list[AgentEvaluation] = []
    total = len(specs)
    if max_workers <= 1:
        for spec in specs:
            if should_cancel and should_cancel():
                raise CancelledError("tournament cancelled")
            ev = evaluate_agent(spec, dev, folds, cfg, bench)
            evaluations.append(ev)
            if on_progress:
                on_progress(len(evaluations), total, ev)
    else:
        payloads = [({"agent_id": s.agent_id, "genome": genome_to_dict(s.genome), "seed": s.seed},
                     dev, folds, cfg, bench) for s in specs]
        with ProcessPoolExecutor(max_workers=max_workers,
                                 mp_context=mp.get_context("spawn")) as pool:
            for ev in pool.map(_evaluate_payload, payloads):
                if should_cancel and should_cancel():
                    pool.shutdown(cancel_futures=True)
                    raise CancelledError("tournament cancelled")
                evaluations.append(ev)
                if on_progress:
                    on_progress(len(evaluations), total, ev)
    ranked = rank_evaluations(evaluations, cfg.ranking)
    bench_curve = stitch(bench, cfg.execution.initial_capital).equity
    return TournamentOutcome(cfg.fingerprint(), [f.to_dict() for f in folds], ranked, bench_curve)


class CancelledError(RuntimeError):
    pass
