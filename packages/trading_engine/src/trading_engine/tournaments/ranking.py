"""Documented, configurable ranking policy (see docs/evaluation.md#ranking).

1. **Hard exclusion**: validation max drawdown above ``max_drawdown_limit``, or a
   failed training run -> status ``excluded`` (ranked last, never selected).
2. **Evidence requirement**: active strategies need ``min_round_trips`` completed
   round trips across all folds; otherwise ``insufficient_evidence`` (ranked after
   all eligible agents, not selectable as parents).
3. **Score** for everyone else::

       score = mean(fold Sharpe)
             - stability_weight * std(fold Sharpe)
             - drawdown_weight  * max drawdown
             - turnover_weight  * annual turnover

   A fold with undefined Sharpe (e.g. never invested) counts as 0.
4. Benchmarks (random, buy-and-hold) are scored and ranked for reference with
   status ``benchmark`` (exempt from exclusion, since they *are* the reference)
   but are never selected for evolution.

Total return is deliberately *not* part of the score.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from trading_engine.tournaments.engine import AgentEvaluation

STATUS_ORDER = {"eligible": 0, "benchmark": 0, "insufficient_evidence": 1, "excluded": 2}


class RankingPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_drawdown_limit: float = Field(0.5, gt=0, le=1)
    min_round_trips: int = Field(5, ge=0)
    stability_weight: float = Field(0.5, ge=0)
    drawdown_weight: float = Field(1.0, ge=0)
    turnover_weight: float = Field(0.01, ge=0)


def score_components(ev: AgentEvaluation, policy: RankingPolicy) -> dict[str, float]:
    sharpes = [m.get("sharpe_ratio") for m in ev.fold_metrics]
    vals = np.array([s if s is not None else 0.0 for s in sharpes], dtype=float)
    mean = float(vals.mean()) if len(vals) else 0.0
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    mdd = float(ev.metrics.get("max_drawdown") or 0.0)
    turnover = float(ev.metrics.get("turnover") or 0.0)
    comps = {
        "mean_fold_sharpe": mean,
        "fold_sharpe_std": std,
        "stability_penalty": -policy.stability_weight * std,
        "drawdown_penalty": -policy.drawdown_weight * mdd,
        "turnover_penalty": -policy.turnover_weight * turnover,
    }
    comps["score"] = (mean + comps["stability_penalty"] + comps["drawdown_penalty"]
                      + comps["turnover_penalty"])
    return comps


def rank_evaluations(
    evaluations: Sequence[AgentEvaluation], policy: RankingPolicy
) -> list[AgentEvaluation]:
    for ev in evaluations:
        if ev.error:
            ev.status, ev.status_reason, ev.score = "excluded", f"failed: {ev.error}", None
            ev.score_components = {}
            continue
        ev.score_components = score_components(ev, policy)
        ev.score = ev.score_components["score"]
        mdd = float(ev.metrics.get("max_drawdown") or 0.0)
        trips = int(ev.metrics.get("n_round_trips") or 0)
        if ev.is_benchmark:
            ev.status, ev.status_reason = "benchmark", "reference strategy; never selected"
        elif mdd > policy.max_drawdown_limit:
            ev.status = "excluded"
            ev.status_reason = (f"max drawdown {mdd:.0%} exceeds limit "
                                f"{policy.max_drawdown_limit:.0%}")
        elif ev.family != "buy_and_hold" and trips < policy.min_round_trips:
            ev.status = "insufficient_evidence"
            ev.status_reason = f"{trips} round trips < required {policy.min_round_trips}"
        else:
            ev.status, ev.status_reason = "eligible", "passed risk and evidence checks"
    ordered = sorted(
        evaluations,
        key=lambda e: (STATUS_ORDER[e.status],
                       -(e.score if e.score is not None else -np.inf), e.agent_id),
    )
    for i, ev in enumerate(ordered, start=1):
        ev.rank = i
    return ordered
