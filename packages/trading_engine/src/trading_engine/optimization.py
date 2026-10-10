"""Optuna hyperparameter optimisation on development data only.

The objective is the tournament ranking score computed on walk-forward
validation folds of :class:`DevelopmentData`; the final test period is never
visible. Fold-by-fold intermediate values (running mean of fold Sharpe) let the
median pruner stop unpromising trials early; the reported objective is the
full ranking score. Budgets are explicit (trial count and wall-clock
timeout) and the sampler is seeded, so a study is reproducible for a fixed
configuration on the same machine.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
from pydantic import BaseModel, ConfigDict, Field

from trading_engine.agents import parse_genome
from trading_engine.agents.genomes import GENOME_TYPES, PositionPolicy, genome_to_dict
from trading_engine.evolution.operators import numeric_genes
from trading_engine.tournaments import AgentSpec, EvaluationConfig
from trading_engine.tournaments.engine import EvaluationAborted, benchmark_folds, evaluate_agent
from trading_engine.tournaments.ranking import rank_evaluations
from trading_engine.training import DevelopmentData, walk_forward_folds

OPTIMISABLE = ("momentum", "random_forest", "xgboost", "mlp")


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    family: str = Field("xgboost", pattern="^(momentum|random_forest|xgboost|mlp)$")
    n_trials: int = Field(20, ge=1, le=100)
    timeout_seconds: float = Field(900, gt=0, le=6 * 3600)
    seed: int = Field(42, ge=0)


@dataclass
class TrialRecord:
    number: int
    state: str
    params: dict[str, Any]
    value: float | None
    fold_scores: list[float]
    duration_seconds: float
    error: str | None = None


@dataclass
class OptimizationResult:
    best_params: dict[str, Any] | None
    best_value: float | None
    best_genome: dict[str, Any] | None
    trials: list[TrialRecord] = field(default_factory=list)


def _suggest(trial: optuna.Trial, family: str) -> dict[str, Any]:
    params: dict[str, Any] = {"family": family}
    for name, (lo, hi, log, is_int) in numeric_genes(GENOME_TYPES[family]).items():
        params[name] = (trial.suggest_int(name, int(lo), int(hi), log=log) if is_int
                        else trial.suggest_float(name, lo, hi, log=log))
    if family == "momentum":
        params["rule"] = trial.suggest_categorical("rule", ["ma_cross", "time_series_momentum",
                                                            "both"])
        if params["fast_window"] >= params["slow_window"]:
            raise optuna.TrialPruned("invalid window ordering")
    else:
        policy = {"threshold": trial.suggest_float("policy.threshold", 0.45, 0.62),
                  "sizing": trial.suggest_categorical("policy.sizing", ["binary", "proportional"])}
        lo, hi, _, _ = numeric_genes(PositionPolicy)["max_allocation"]
        policy["max_allocation"] = trial.suggest_float("policy.max_allocation", lo, hi)
        params["policy"] = policy
    if family == "mlp":
        width = trial.suggest_categorical("hidden_width", [8, 16, 32, 64])
        depth = trial.suggest_int("hidden_depth", 1, 2)
        params["hidden_layers"] = [width] * depth
        params["max_epochs"] = min(params["max_epochs"], 120)
    return params


def optimize(
    dev: DevelopmentData,
    evaluation: EvaluationConfig,
    config: OptimizationConfig,
    *,
    on_trial: Callable[[TrialRecord], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> OptimizationResult:
    if not isinstance(dev, DevelopmentData):
        raise TypeError("optimisation only accepts DevelopmentData")
    folds = walk_forward_folds(dev, evaluation.walk_forward)
    bench = benchmark_folds(dev, folds, evaluation)
    records: list[TrialRecord] = []

    def objective(trial: optuna.Trial) -> float:
        if should_cancel and should_cancel():
            trial.study.stop()
            raise optuna.TrialPruned("cancelled")
        started = time.perf_counter()
        params = _suggest(trial, config.family)
        genome = parse_genome(params)
        fold_scores: list[float] = []

        def on_fold(k: int, metrics: dict[str, Any]) -> None:
            sharpe = metrics.get("sharpe_ratio")
            fold_scores.append(float(sharpe) if sharpe is not None else 0.0)
            if k < len(folds) - 1:
                trial.report(float(np.mean(fold_scores)), step=k)
                if trial.should_prune():
                    raise EvaluationAborted

        try:
            ev = evaluate_agent(AgentSpec(f"trial-{trial.number}", genome, config.seed),
                                dev, folds, evaluation, bench, on_fold=on_fold)
        except EvaluationAborted:
            _record(trial, "pruned", None, fold_scores, started, genome)
            raise optuna.TrialPruned() from None
        if ev.error:
            _record(trial, "failed", None, fold_scores, started, genome, ev.error)
            raise RuntimeError(ev.error)
        ranked = rank_evaluations([ev], evaluation.ranking)[0]
        value = float(ranked.score if ranked.score is not None else -10.0)
        _record(trial, "complete", value, fold_scores, started, genome)
        return value

    def _record(trial: optuna.Trial, state: str, value: float | None, scores: list[float],
                started: float, genome: Any, error: str | None = None) -> None:
        rec = TrialRecord(trial.number, state, genome_to_dict(genome), value,
                          [round(s, 6) for s in scores], round(time.perf_counter() - started, 3),
                          error)
        records.append(rec)
        if on_trial:
            on_trial(rec)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=config.n_trials, timeout=config.timeout_seconds,
                   catch=(RuntimeError, ValueError), n_jobs=1)
    recorded = {r.number for r in records}
    for t in study.trials:  # trials that ended before evaluation (invalid params, errors)
        if t.number in recorded:
            continue
        state = "pruned" if t.state == optuna.trial.TrialState.PRUNED else "failed"
        reason = "invalid parameter combination" if state == "pruned" else str(
            t.system_attrs.get("fail_reason", "failed"))
        records.append(TrialRecord(t.number, state, dict(t.params), None, [], 0.0, reason))
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return OptimizationResult(None, None, None, sorted(records, key=lambda r: r.number))
    best = study.best_trial
    best_rec = next(r for r in records if r.number == best.number)
    return OptimizationResult(dict(best.params), float(np.round(best.value, 6)),
                              best_rec.params, sorted(records, key=lambda r: r.number))
