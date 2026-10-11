"""Long-only, single-asset, event-driven daily simulator.

Event order for each session ``i``::

    open(i):  fill the order submitted after close(i-1)        (EXECUTION)
    close(i): mark the portfolio to market, append equity row  (VALUATION)
    close(i): read the agent's target weight for day i,
              pass it through the RiskManager, and if it changes the
              position enough, submit an order for open(i+1)     (DECISION)

So a decision can never fill at the price that was used to make it. The target
weights are supplied up front as an array aligned with the bars; the agents
produce them from point-in-time features, and the simulator only ever uses
``weights[i]`` to trade at ``open[i+1]``.

All amounts are simulated. There is no broker connection anywhere in AETHER.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from trading_engine.risk import PortfolioSnapshot, RiskConfig, RiskManager

SIGNAL_TIME = pd.Timedelta(hours=16)  # nominal exchange-local close
EXECUTION_TIME = pd.Timedelta(hours=9, minutes=30)  # nominal exchange-local open


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    initial_capital: float = Field(10_000.0, gt=0, le=1e9)
    fee_bps: float = Field(1.0, ge=0, le=500)
    fee_fixed: float = Field(0.0, ge=0, le=100)
    slippage_bps: float = Field(5.0, ge=0, le=500)
    fractional_shares: bool = True
    min_rebalance_weight: float = Field(0.02, ge=0, le=0.5)


@dataclass(frozen=True)
class Trade:
    signal_time: str
    submitted_time: str
    executed_time: str
    side: str
    quantity: float
    reference_price: float
    fill_price: float
    notional: float
    fee: float
    slippage_cost: float
    target_weight: float
    position_after: float
    cash_after: float


@dataclass(frozen=True)
class RejectedOrder:
    time: str
    target_weight: float
    reason: str


@dataclass
class SimulationResult:
    equity: pd.DataFrame
    trades: list[Trade]
    rejected_orders: list[RejectedOrder]
    risk_events: list[dict[str, object]]
    config: ExecutionConfig
    risk: RiskConfig
    stale_price_days: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def final_equity(self) -> float:
        return float(self.equity["equity"].iloc[-1]) if len(self.equity) else math.nan

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(t) for t in self.trades])


def _fmt(ts: pd.Timestamp, offset: pd.Timedelta) -> str:
    return (ts.normalize() + offset).isoformat()


# "THH:MM:SS" suffixes of the nominal event times, exactly as isoformat() renders them.
_SIGNAL_SUFFIX = (pd.Timestamp(0) + SIGNAL_TIME).isoformat()[10:]
_EXECUTION_SUFFIX = (pd.Timestamp(0) + EXECUTION_TIME).isoformat()[10:]


class _Stamps:
    """ISO timestamps for signal/execution events, formatted once per bar.

    Formatting each trade's timestamps through pandas boxing dominated the
    loop for trade-heavy agents. For tz-naive indexes (all AETHER datasets) the
    session dates are formatted in one vectorised call; the strings are
    identical to ``_fmt``. Tz-aware indexes fall back to ``_fmt``.
    """

    def __init__(self, dates: pd.DatetimeIndex) -> None:
        self.dates = dates
        self.days = (
            np.datetime_as_string(dates.normalize().to_numpy(), unit="D")
            if dates.tz is None
            else None
        )

    def signal(self, i: int) -> str:
        if self.days is None:
            return _fmt(self.dates[i], SIGNAL_TIME)
        return f"{self.days[i]}{_SIGNAL_SUFFIX}"

    def execution(self, i: int) -> str:
        if self.days is None:
            return _fmt(self.dates[i], EXECUTION_TIME)
        return f"{self.days[i]}{_EXECUTION_SUFFIX}"


def simulate(
    bars: pd.DataFrame,
    target_weights: np.ndarray | pd.Series,
    *,
    execution: ExecutionConfig | None = None,
    risk: RiskConfig | None = None,
    realised_vol: pd.Series | None = None,
) -> SimulationResult:
    """Simulate one agent on one symbol.

    ``bars`` must be indexed by session date (ascending) with open/close columns;
    ``target_weights[i]`` is the decision taken after the close of ``bars[i]``.
    """
    cfg = execution or ExecutionConfig()
    risk_cfg = risk or RiskConfig()
    weights = np.asarray(target_weights, dtype="float64")
    if len(weights) != len(bars):
        raise ValueError("target_weights must align with bars")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("bars must be sorted")
    manager = RiskManager(risk_cfg)
    n = len(bars)
    opens = bars["open"].to_numpy(dtype="float64")
    closes = bars["close"].to_numpy(dtype="float64")
    dates = pd.DatetimeIndex(bars.index)
    stamps = _Stamps(dates)
    vols = (
        realised_vol.reindex(bars.index).to_numpy(dtype="float64")
        if realised_vol is not None
        else np.full(n, np.nan)
    )
    fee_rate = cfg.fee_bps / 1e4
    slip = cfg.slippage_bps / 1e4

    cash = cfg.initial_capital
    qty = 0.0
    peak = cfg.initial_capital
    last_close = math.nan
    pending: tuple[float, int] | None = None  # (target weight, decision index)
    trades: list[Trade] = []
    rejected: list[RejectedOrder] = []
    risk_events: list[dict[str, object]] = []
    rows: list[tuple[float, ...]] = []
    stale_days = 0

    for i in range(n):
        # ---- EXECUTION at open(i) -------------------------------------------------
        if pending is not None:
            target, decided_at = pending
            pending = None
            ref = opens[i]
            if not math.isfinite(ref) or ref <= 0:
                rejected.append(RejectedOrder(stamps.execution(i), target, "missing_market_data"))
            else:
                trade, reason = _execute(
                    cash, qty, target, ref, fee_rate, cfg.fee_fixed, slip, cfg.fractional_shares
                )
                if trade is None:
                    rejected.append(
                        RejectedOrder(stamps.execution(i), target, reason or "rejected")
                    )
                else:
                    side, dq, fill, fee = trade
                    notional = abs(dq) * fill
                    cash_before = cash
                    if side == "buy":
                        cash -= notional + fee
                        qty += dq
                    else:
                        cash += notional - fee
                        qty -= dq
                    if qty < 1e-12:
                        qty = 0.0
                    # Going all-in spends cash to ~0 and leaves float residue that scales
                    # with the amount (~1e-16 x cash). The tolerance must be relative:
                    # a fixed 1e-9 failed correct trades once equity grew past ~1e7.
                    if cash < 0 and cash > -1e-9 * max(1.0, cash_before):
                        cash = 0.0
                    if cash < 0:
                        raise AssertionError("simulator produced negative cash")
                    signal_time = stamps.signal(decided_at)
                    trades.append(
                        Trade(
                            signal_time=signal_time,
                            submitted_time=signal_time,
                            executed_time=stamps.execution(i),
                            side=side,
                            quantity=dq,
                            reference_price=ref,
                            fill_price=fill,
                            notional=notional,
                            fee=fee,
                            slippage_cost=abs(fill - ref) * dq,
                            target_weight=target,
                            position_after=qty,
                            cash_after=cash,
                        )
                    )

        # ---- VALUATION at close(i) ------------------------------------------------
        px = closes[i]
        if math.isfinite(px) and px > 0:
            last_close = px
        else:
            stale_days += 1
        mark = (
            last_close
            if math.isfinite(last_close)
            else (opens[i] if math.isfinite(opens[i]) else 0.0)
        )
        position_value = qty * mark
        equity = cash + position_value
        peak = max(peak, equity)
        weight = position_value / equity if equity > 0 else 0.0
        rows.append((cash, qty, mark, position_value, equity, weight, equity / peak - 1))

        # ---- DECISION after close(i) ---------------------------------------------
        if i == n - 1:
            break  # no next session to execute in
        snapshot = PortfolioSnapshot(
            equity=equity,
            peak_equity=peak,
            current_weight=weight,
            realised_vol=vols[i] if math.isfinite(vols[i]) else None,
        )
        decision = manager.review(float(weights[i]), snapshot)
        for ev in decision.events:
            risk_events.append(
                {
                    "time": stamps.signal(i),
                    "code": ev.code,
                    "message": ev.message,
                    "proposed_weight": ev.proposed_weight,
                    "approved_weight": ev.approved_weight,
                }
            )
        target = decision.approved_weight
        full_exit = target == 0.0 and qty > 0
        rebalance = full_exit or abs(target - weight) >= max(cfg.min_rebalance_weight, 1e-9)
        if rebalance and not (target == 0.0 and qty == 0.0):
            pending = (target, i)

    equity_frame = pd.DataFrame(
        rows,
        columns=["cash", "quantity", "price", "position_value", "equity", "weight", "drawdown"],
        index=dates[: len(rows)],
    )
    equity_frame.index.name = "timestamp"
    return SimulationResult(
        equity_frame, trades, rejected, risk_events, cfg, risk_cfg, stale_price_days=stale_days
    )


def _execute(
    cash: float,
    qty: float,
    target: float,
    ref: float,
    fee_rate: float,
    fee_fixed: float,
    slip: float,
    fractional: bool,
) -> tuple[tuple[str, float, float, float] | None, str | None]:
    """Size an order at the open. Returns ((side, qty, fill, fee), None) or (None, reason)."""
    equity_open = cash + qty * ref
    target_qty = target * equity_open / ref
    if target_qty > qty:
        fill = ref * (1 + slip)
        want = target_qty - qty
        affordable = (cash - fee_fixed) / (fill * (1 + fee_rate))
        dq = min(want, affordable)
        if not fractional:
            dq = math.floor(dq)
        if dq <= 1e-12:
            return None, "insufficient_cash" if affordable < want else "size_rounds_to_zero"
        return ("buy", dq, fill, fee_fixed + fee_rate * dq * fill), None
    fill = ref * (1 - slip)
    dq = qty if target <= 0 else min(qty, qty - target_qty)
    if not fractional and target > 0:
        dq = math.floor(dq)
    if dq <= 1e-12:
        return None, "size_rounds_to_zero"
    fee = fee_fixed + fee_rate * dq * fill
    if dq * fill - fee < 0 and cash + dq * fill - fee < 0:
        return None, "fees_exceed_proceeds"
    return ("sell", dq, fill, fee), None
