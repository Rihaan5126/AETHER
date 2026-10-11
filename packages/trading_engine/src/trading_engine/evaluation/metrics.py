"""Performance metrics computed from simulated equity curves and trade ledgers.

Conventions (see docs/evaluation.md):

* Daily simple returns of total equity (cash + marked position), after all fees
  and slippage. Risk-free rate is 0.
* Annualisation uses 252 sessions.
* Undefined quantities are ``None`` rather than ``inf``/``nan``: e.g. Sharpe when
  volatility is zero, profit factor without losing trades, win rate without any
  completed round trip, ratios on fewer than ``MIN_SESSIONS`` sessions.
* ``max_drawdown`` is reported as a positive fraction (0.25 = -25% from peak).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from trading_engine.simulation import SimulationResult
from trading_engine.timing import TRADING_DAYS_PER_YEAR

MIN_SESSIONS = 20
_EPS = 1e-12


@dataclass(frozen=True)
class RoundTrip:
    entry_time: str
    exit_time: str
    pnl: float
    return_pct: float
    holding_sessions: int


@dataclass(frozen=True)
class PerformanceMetrics:
    sessions: int
    total_return: float
    annualized_return: float | None
    annualized_volatility: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    max_drawdown: float
    calmar_ratio: float | None
    win_rate: float | None
    profit_factor: float | None
    n_trades: int
    n_round_trips: int
    turnover: float | None
    exposure: float
    avg_holding_period: float | None
    total_fees: float
    total_slippage: float
    cost_drag: float
    benchmark_return: float | None
    excess_return: float | None
    rejected_orders: int
    risk_events: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {k: _clean(v) for k, v in asdict(self).items()}


def _clean(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def round_trips(result: SimulationResult) -> list[RoundTrip]:
    """Completed flat -> long -> flat episodes with net P&L after costs."""
    eq_index = pd.DatetimeIndex(result.equity.index)
    if eq_index.tz is None:
        # Trade times are ISO strings that start with the session date; one
        # vectorised format beats boxing every equity timestamp.
        days = np.datetime_as_string(eq_index.normalize().to_numpy(), unit="D")
        by_day = {d: i for i, d in enumerate(days.tolist())}

        def position(stamp: str, default: int) -> int:
            return by_day.get(stamp[:10], default)
    else:
        by_ts = {ts.normalize(): i for i, ts in enumerate(eq_index)}

        def position(stamp: str, default: int) -> int:
            return by_ts.get(pd.Timestamp(stamp).normalize(), default)

    trips: list[RoundTrip] = []
    flow = 0.0
    invested = 0.0
    entry: str | None = None
    for t in result.trades:
        if t.side == "buy":
            if entry is None:
                entry, flow, invested = t.executed_time, 0.0, 0.0
            flow -= t.notional + t.fee
            invested += t.notional + t.fee
        else:
            flow += t.notional - t.fee
        if entry is not None and t.position_after == 0.0:
            start = position(entry, 0)
            stop = position(t.executed_time, start)
            trips.append(
                RoundTrip(
                    entry,
                    t.executed_time,
                    flow,
                    flow / invested if invested > 0 else 0.0,
                    stop - start,
                )
            )
            entry = None
    return trips


def compute_metrics(
    result: SimulationResult,
    benchmark_equity: pd.Series | None = None,
    *,
    trips: list[RoundTrip] | None = None,
) -> PerformanceMetrics:
    """Metrics for one simulation. ``trips`` overrides round-trip detection (used when
    pooling independently simulated folds)."""
    eq = result.equity["equity"]
    n = len(eq)
    initial = result.config.initial_capital
    total_return = float(eq.iloc[-1] / initial - 1) if n else 0.0
    rets = eq.pct_change().fillna(eq.iloc[0] / initial - 1 if n else 0.0)
    years = n / TRADING_DAYS_PER_YEAR
    enough = n >= MIN_SESSIONS

    ann_ret = (1 + total_return) ** (1 / years) - 1 if enough and total_return > -1 else None
    std = float(rets.std(ddof=1)) if n > 1 else 0.0
    ann_vol = std * math.sqrt(TRADING_DAYS_PER_YEAR) if enough else None
    sharpe = (
        float(rets.mean()) / std * math.sqrt(TRADING_DAYS_PER_YEAR)
        if enough and std > _EPS
        else None
    )
    downside = np.minimum(rets.to_numpy(), 0.0)
    dd_dev = math.sqrt(float(np.mean(downside**2))) if n else 0.0
    sortino = (
        float(rets.mean()) / dd_dev * math.sqrt(TRADING_DAYS_PER_YEAR)
        if enough and dd_dev > _EPS
        else None
    )
    mdd = float(-drawdown_series(pd.concat([pd.Series([initial]), eq])).min()) if n else 0.0
    calmar = ann_ret / mdd if ann_ret is not None and mdd > _EPS else None

    if trips is None:
        trips = round_trips(result)
    wins = [t for t in trips if t.pnl > 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in trips if t.pnl < 0)
    win_rate = len(wins) / len(trips) if trips else None
    profit_factor = gross_win / gross_loss if gross_loss > _EPS else None
    avg_hold = float(np.mean([t.holding_sessions for t in trips])) if trips else None

    traded = sum(t.notional for t in result.trades)
    mean_eq = float(eq.mean()) if n else initial
    turnover = traded / mean_eq / years if years > 0 and mean_eq > 0 and enough else None
    fees = sum(t.fee for t in result.trades)
    slippage = sum(t.slippage_cost for t in result.trades)

    bench_ret = None
    if benchmark_equity is not None and len(benchmark_equity):
        bench_ret = float(benchmark_equity.iloc[-1] / initial - 1)
    return PerformanceMetrics(
        sessions=n,
        total_return=total_return,
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=mdd,
        calmar_ratio=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        n_trades=len(result.trades),
        n_round_trips=len(trips),
        turnover=turnover,
        exposure=float(result.equity["weight"].mean()) if n else 0.0,
        avg_holding_period=avg_hold,
        total_fees=fees,
        total_slippage=slippage,
        cost_drag=(fees + slippage) / initial,
        benchmark_return=bench_ret,
        excess_return=total_return - bench_ret if bench_ret is not None else None,
        rejected_orders=len(result.rejected_orders),
        risk_events=len(result.risk_events),
    )
