from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from aether_api.schemas.common import APIModel, UTCDateTime


class LeaderboardRow(BaseModel):
    rank: int | None
    agent_id: str
    agent_name: str
    algorithm: str
    generation: int
    is_benchmark: bool
    status: str
    status_reason: str
    score: float | None
    score_components: dict[str, float]
    total_return: float | None
    annualized_return: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    max_drawdown: float | None
    turnover: float | None
    n_trades: int | None
    n_round_trips: int | None
    win_rate: float | None
    exposure: float | None
    excess_return: float | None
    stability: float | None
    warnings: list[dict[str, Any]]


class TournamentOut(APIModel):
    id: str
    experiment_id: str
    experiment_name: str
    generation: int
    period: str
    status: str
    config_fingerprint: str
    n_agents: int
    created_at: UTCDateTime
    completed_at: UTCDateTime | None


class TournamentDetail(TournamentOut):
    evaluation_config: dict[str, Any]
    folds: list[dict[str, Any]]
    benchmark_metrics: dict[str, Any] | None


class Leaderboard(BaseModel):
    tournament_id: str
    experiment_id: str
    generation: int
    period: Literal["validation"]
    benchmark: dict[str, Any] | None
    ranking_policy: dict[str, Any]
    rows: list[LeaderboardRow]


class EquitySeries(BaseModel):
    agent_id: str
    points: list[dict[str, Any]]


class EquityCurves(BaseModel):
    tournament_id: str
    period: str
    downsampled: bool
    series: list[EquitySeries]
    benchmark: EquitySeries | None


class TradeOut(APIModel):
    id: int
    agent_id: str
    symbol: str
    timestamp: datetime
    signal_time: datetime
    side: str
    quantity: float
    execution_price: float
    reference_price: float
    fees: float
    slippage: float
    notional: float
    target_weight: float


class AgentChoice(BaseModel):
    family: Literal["random", "buy_and_hold", "momentum", "random_forest", "xgboost", "mlp"]
    genome: dict[str, Any] | None = None
    seed: int | None = Field(None, ge=0, le=2**31 - 1)


class TournamentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    dataset_id: str
    symbol: str
    agents: list[AgentChoice] = Field(..., min_length=1, max_length=24)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    initial_capital: float = Field(10_000, gt=0, le=1e9)
    fee_bps: float = Field(1.0, ge=0, le=500)
    slippage_bps: float = Field(5.0, ge=0, le=500)
    n_folds: int = Field(3, ge=1, le=8)
    test_fraction: float = Field(0.2, gt=0.05, lt=0.5)
