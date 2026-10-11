"""Statistical warnings attached to every evaluated agent.

Warnings never change scores by themselves; they explain why a result should not
be trusted. The ranking policy separately applies hard exclusions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from trading_engine.evaluation.metrics import PerformanceMetrics


class WarningThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_round_trips: int = Field(10, ge=1)
    high_drawdown: float = Field(0.30, gt=0, lt=1)
    excessive_turnover: float = Field(25.0, gt=0)
    unstable_fold_sharpe_std: float = Field(1.0, gt=0)
    single_period_share: float = Field(0.8, gt=0, le=1)
    # Daily-bar strategies with a validation Sharpe this high are rare; on thin
    # evidence it far more often signals luck, leakage or a bug than a real edge.
    suspicious_sharpe: float = Field(2.0, gt=0)
    suspicious_max_round_trips: int = Field(30, ge=1)
    suspicious_max_years: float = Field(2.0, gt=0)
    overfit_sharpe_gap: float = Field(1.0, gt=0)


def _w(code: str, severity: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def statistical_warnings(
    overall: PerformanceMetrics,
    folds: Sequence[PerformanceMetrics],
    *,
    is_active: bool,
    thresholds: WarningThresholds | None = None,
    train_sharpe: float | None = None,
) -> list[dict[str, str]]:
    t = thresholds or WarningThresholds()
    out: list[dict[str, str]] = []
    if is_active and overall.n_round_trips < t.min_round_trips:
        out.append(
            _w(
                "small_sample",
                "high",
                f"only {overall.n_round_trips} completed round trips; results are "
                "statistically weak",
            )
        )
    years = overall.sessions / 252
    if (
        is_active
        and overall.sharpe_ratio is not None
        and overall.sharpe_ratio >= t.suspicious_sharpe
        and (overall.n_round_trips < t.suspicious_max_round_trips or years < t.suspicious_max_years)
    ):
        out.append(
            _w(
                "suspiciously_strong",
                "high",
                f"validation Sharpe {overall.sharpe_ratio:.2f} from {overall.n_round_trips} round "
                f"trips over {years:.1f} years; results this strong on little evidence are "
                "usually luck or an error",
            )
        )
    if (
        is_active
        and train_sharpe is not None
        and overall.sharpe_ratio is not None
        and train_sharpe - overall.sharpe_ratio > t.overfit_sharpe_gap
    ):
        out.append(
            _w(
                "overfit_gap",
                "medium",
                f"in-sample Sharpe {train_sharpe:.2f} vs validation {overall.sharpe_ratio:.2f}; "
                "the model fits its training data far better than unseen data",
            )
        )
    if overall.max_drawdown >= t.high_drawdown:
        out.append(_w("high_drawdown", "high", f"maximum drawdown of {overall.max_drawdown:.0%}"))
    if overall.turnover is not None and overall.turnover > t.excessive_turnover:
        out.append(
            _w(
                "excessive_turnover",
                "medium",
                f"annual turnover {overall.turnover:.1f}x equity; highly cost-sensitive",
            )
        )
    gross = overall.total_return + overall.cost_drag
    if gross > 0 and overall.total_return <= 0:
        out.append(
            _w(
                "negative_after_costs",
                "high",
                "profitable before costs but not after fees and slippage",
            )
        )
    elif gross > 0 and overall.cost_drag > 0.5 * gross:
        out.append(
            _w("cost_dominated", "medium", "costs consumed more than half of the gross gain")
        )
    if overall.excess_return is not None and overall.excess_return < 0 and is_active:
        out.append(
            _w("below_benchmark", "info", f"trailed buy-and-hold by {-overall.excess_return:.1%}")
        )
    sharpes = [f.sharpe_ratio for f in folds if f.sharpe_ratio is not None]
    if len(sharpes) >= 2:
        std = float(np.std(sharpes, ddof=1))
        if std > t.unstable_fold_sharpe_std or (min(sharpes) < 0 < max(sharpes)):
            out.append(
                _w(
                    "unstable_folds",
                    "medium",
                    f"fold Sharpe ranges {min(sharpes):.2f} to {max(sharpes):.2f}",
                )
            )
    gains = [max(f.total_return, 0.0) for f in folds]
    if len(folds) >= 2 and sum(gains) > 0 and max(gains) / sum(gains) >= t.single_period_share:
        out.append(
            _w(
                "single_period_dependence",
                "medium",
                "most of the positive return came from a single validation window",
            )
        )
    return out
