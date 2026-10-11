"""Centralised risk controls applied by the simulator to every agent equally."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RiskConfig(BaseModel):
    """Hard limits. Agents cannot change these; the simulator enforces them.

    * ``max_position``: cap on the target weight (no leverage: <= 1).
    * ``min_cash_fraction``: weight is capped at ``1 - min_cash_fraction``.
    * ``max_daily_turnover``: largest weight change per session; larger moves are
      executed partially, step by step.
    * ``drawdown_circuit_breaker``: when equity falls this far below its peak the
      strategy is forced flat for ``breaker_cooldown_bars`` sessions.
    * ``volatility_target``: optional annualised volatility target; weights are
      scaled by ``min(1, target / realised_vol_20)``.
    """

    model_config = ConfigDict(frozen=True)

    max_position: float = Field(1.0, gt=0, le=1.0)
    min_cash_fraction: float = Field(0.0, ge=0, lt=1)
    max_daily_turnover: float | None = Field(None, gt=0, le=2)
    drawdown_circuit_breaker: float | None = Field(None, gt=0, lt=1)
    breaker_cooldown_bars: int = Field(20, ge=1, le=252)
    volatility_target: float | None = Field(None, gt=0, le=2)
    allow_short: bool = False
    allow_leverage: bool = False

    @field_validator("allow_short", "allow_leverage")
    @classmethod
    def _mvp_long_only(cls, value: bool) -> bool:
        if value:
            raise ValueError("short selling and leverage are not supported in the MVP")
        return value


@dataclass(frozen=True)
class RiskEvent:
    code: str
    message: str
    proposed_weight: float | None
    approved_weight: float


@dataclass
class PortfolioSnapshot:
    equity: float
    peak_equity: float
    current_weight: float
    realised_vol: float | None = None


@dataclass
class RiskDecision:
    approved_weight: float
    events: list[RiskEvent] = field(default_factory=list)


class RiskManager:
    """Stateful reviewer of target weights (one instance per simulation)."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._breaker_remaining = 0

    @property
    def breaker_active(self) -> bool:
        return self._breaker_remaining > 0

    def review(self, proposed: float, state: PortfolioSnapshot) -> RiskDecision:
        cfg = self.config
        events: list[RiskEvent] = []

        def note(code: str, msg: str, approved: float) -> None:
            events.append(
                RiskEvent(code, msg, proposed if math.isfinite(proposed) else None, approved)
            )

        if self._breaker_remaining > 0:
            self._breaker_remaining -= 1
            note("circuit_breaker_active", "drawdown circuit breaker: holding cash", 0.0)
            return RiskDecision(0.0, events)

        if cfg.drawdown_circuit_breaker is not None and state.peak_equity > 0:
            dd = state.equity / state.peak_equity - 1
            if dd <= -cfg.drawdown_circuit_breaker:
                self._breaker_remaining = cfg.breaker_cooldown_bars
                note("circuit_breaker_tripped", f"drawdown {dd:.1%} breached limit", 0.0)
                return RiskDecision(0.0, events)

        if not math.isfinite(proposed):
            note(
                "invalid_signal",
                "non-finite target weight rejected; position unchanged",
                state.current_weight,
            )
            return RiskDecision(state.current_weight, events)

        weight = proposed
        if weight < 0:
            weight = 0.0
            note("short_not_allowed", "negative weight clipped to 0 (long-only)", weight)
        cap = min(cfg.max_position, 1.0 - cfg.min_cash_fraction)
        if weight > cap + 1e-12:
            weight = cap
            note("position_limit", f"weight clipped to {cap:.2f}", weight)

        if cfg.volatility_target is not None and state.realised_vol and state.realised_vol > 0:
            scale = min(1.0, cfg.volatility_target / state.realised_vol)
            if scale < 1.0 and weight > 0:
                weight *= scale
                note("volatility_scaling", f"scaled by {scale:.2f}", weight)

        if cfg.max_daily_turnover is not None:
            delta = weight - state.current_weight
            if abs(delta) > cfg.max_daily_turnover:
                weight = state.current_weight + math.copysign(cfg.max_daily_turnover, delta)
                note("turnover_limit", "weight change limited by max_daily_turnover", weight)
        return RiskDecision(weight, events)
