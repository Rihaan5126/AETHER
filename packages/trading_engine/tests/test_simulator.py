from __future__ import annotations

import math

import numpy as np
import pytest
from bar_factory import make_bars
from trading_engine.evaluation import compute_metrics, round_trips
from trading_engine.risk import RiskConfig
from trading_engine.simulation import ExecutionConfig, simulate

NO_COST = ExecutionConfig(fee_bps=0, slippage_bps=0, min_rebalance_weight=0)


def test_no_trade_keeps_cash():
    bars = make_bars([100, 101, 99, 102])
    res = simulate(bars, np.zeros(4), execution=NO_COST)
    assert res.trades == [] and np.allclose(res.equity["equity"], 10_000)


def test_buy_and_hold_executes_next_open_and_tracks_close():
    bars = make_bars([100, 110, 120, 130], closes=[105, 115, 125, 135])
    res = simulate(bars, np.ones(4), execution=NO_COST)
    t = res.trades[0]
    assert t.executed_time.startswith(str(bars.index[1].date()))  # decided day 0, filled day 1
    assert t.signal_time.startswith(str(bars.index[0].date()))
    assert t.fill_price == 110 and math.isclose(t.quantity, 10_000 / 110)
    assert res.equity["equity"].iloc[0] == 10_000  # nothing happens on the decision day
    assert math.isclose(res.equity["equity"].iloc[-1], 10_000 / 110 * 135)
    assert len(res.trades) == 1


def test_no_lookahead_fill_price():
    bars = make_bars([100, 50, 200], closes=[100, 50, 200])
    res = simulate(bars, np.array([1.0, 0.0, 0.0]), execution=NO_COST)
    # Decision on day 0 can only fill at day-1 open (50), never at day-0 close (100).
    assert res.trades[0].fill_price == 50
    assert res.trades[1].fill_price == 200 and res.trades[1].side == "sell"


def test_fees_and_slippage_are_exact():
    cfg = ExecutionConfig(fee_bps=10, fee_fixed=1.0, slippage_bps=50, min_rebalance_weight=0)
    bars = make_bars([100, 100, 100, 100])
    res = simulate(bars, np.array([1.0, 0.0, 0.0, 0.0]), execution=cfg)
    buy, sell = res.trades
    fill = 100 * 1.005
    qty = (10_000 - 1.0) / (fill * 1.001)
    assert math.isclose(buy.fill_price, fill) and math.isclose(buy.quantity, qty)
    assert math.isclose(buy.fee, 1.0 + 0.001 * qty * fill)
    cash_after_buy = 10_000 - qty * fill - buy.fee
    assert math.isclose(buy.cash_after, cash_after_buy, abs_tol=1e-9)
    assert math.isclose(sell.fill_price, 99.5)
    proceeds = qty * 99.5
    final = cash_after_buy + proceeds - (1.0 + 0.001 * proceeds)
    assert math.isclose(res.final_equity, final)
    trips = round_trips(res)
    assert len(trips) == 1 and math.isclose(trips[0].pnl, final - 10_000)


def test_cash_conservation_and_no_negative_cash():
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
    bars = make_bars(prices * (1 + rng.normal(0, 0.003, 300)), prices)
    weights = rng.choice([0.0, 0.3, 0.7, 1.0], size=300)
    cfg = ExecutionConfig(fee_bps=5, fee_fixed=0.5, slippage_bps=10)
    res = simulate(bars, weights, execution=cfg)
    cash = 10_000.0
    for t in res.trades:
        cash += (-1 if t.side == "buy" else 1) * t.notional - t.fee
        assert math.isclose(cash, t.cash_after, rel_tol=1e-9, abs_tol=1e-6)
    assert (res.equity["cash"] >= -1e-9).all()
    eq = res.equity
    assert np.allclose(eq["equity"], eq["cash"] + eq["quantity"] * eq["price"])
    assert (eq["weight"] <= 1 + 1e-9).all()


def test_partial_allocation():
    bars = make_bars([50, 50, 50, 60], closes=[50, 50, 50, 60])
    res = simulate(bars, np.full(4, 0.5), execution=NO_COST)
    assert math.isclose(res.equity["weight"].iloc[1], 0.5)
    assert math.isclose(res.final_equity, 5_000 + 100 * 60)


def test_integer_share_sizing():
    cfg = NO_COST.model_copy(update={"fractional_shares": False})
    bars = make_bars([333, 333, 333])
    res = simulate(bars, np.ones(3), execution=cfg)
    assert res.trades[0].quantity == 30 and math.isclose(res.equity["cash"].iloc[-1], 10)


def test_missing_open_rejects_order_and_stale_close_is_reported():
    bars = make_bars([100, np.nan, 100, 100], closes=[100, np.nan, 100, 100])
    res = simulate(bars, np.array([1.0, 1.0, 1.0, 1.0]), execution=NO_COST)
    assert res.rejected_orders[0].reason == "missing_market_data"
    assert res.trades[0].executed_time.startswith(str(bars.index[2].date()))
    assert res.stale_price_days == 1


def test_risk_layer_clips_and_rejects():
    bars = make_bars([100] * 6)
    weights = np.array([1.7, -0.5, np.nan, 0.5, 0.5, 0.5])
    res = simulate(bars, weights, execution=NO_COST, risk=RiskConfig(max_position=0.8))
    codes = [e["code"] for e in res.risk_events]
    assert codes[:3] == ["position_limit", "short_not_allowed", "invalid_signal"]
    assert math.isclose(res.trades[0].target_weight, 0.8)
    assert (res.equity["weight"] <= 0.8 + 1e-9).all()


def test_turnover_limit_and_min_cash():
    bars = make_bars([100] * 5)
    res = simulate(
        bars,
        np.ones(5),
        execution=NO_COST,
        risk=RiskConfig(max_daily_turnover=0.25, min_cash_fraction=0.1),
    )
    assert [round(t.target_weight, 2) for t in res.trades] == [0.25, 0.5, 0.75, 0.9]


def test_drawdown_circuit_breaker_forces_cash():
    prices = [100, 100, 60, 60, 60, 60]
    bars = make_bars(prices, prices)
    res = simulate(
        bars,
        np.ones(6),
        execution=NO_COST,
        risk=RiskConfig(drawdown_circuit_breaker=0.3, breaker_cooldown_bars=2),
    )
    assert any(e["code"] == "circuit_breaker_tripped" for e in res.risk_events)
    assert res.trades[-1].side == "sell"


def test_leverage_and_short_cannot_be_enabled():
    with pytest.raises(ValueError):
        RiskConfig(allow_short=True)
    with pytest.raises(ValueError):
        RiskConfig(max_position=1.5)


def test_simulation_is_deterministic():
    rng = np.random.default_rng(3)
    prices = 100 + np.cumsum(rng.normal(0, 1, 100))
    bars = make_bars(prices, prices)
    w = rng.random(100)
    a, b = simulate(bars, w), simulate(bars, w)
    assert a.equity.equals(b.equity) and a.trades == b.trades


def test_metrics_undefined_cases_and_known_values():
    bars = make_bars([100] * 30)
    flat = compute_metrics(simulate(bars, np.zeros(30)))
    assert flat.sharpe_ratio is None and flat.sortino_ratio is None
    assert flat.calmar_ratio is None and flat.win_rate is None and flat.profit_factor is None
    prices = [100, 100, 120, 90, 90] + [90] * 25
    bars = make_bars(prices, prices)
    m = compute_metrics(simulate(bars, np.ones(30), execution=NO_COST))
    assert math.isclose(m.max_drawdown, 0.25)
    assert math.isclose(m.total_return, -0.10)
    assert (
        m.n_trades == 1 and m.n_round_trips == 0 and math.isclose(m.exposure, 29 / 30, rel_tol=1e-9)
    )


def test_full_allocation_at_large_equity_never_trips_cash_check():
    """Regression: the negative-cash guard used an absolute 1e-9 tolerance.

    Going all-in spends cash down to ~0, so float rounding leaves a residue on the
    order of 1e-16 * equity. Once compounding pushed equity past ~1e7 (seen on
    long in-sample windows), that residue exceeded 1e-9 and the simulator raised
    "negative cash" on a correct trade, so the agent was marked failed.
    """
    rng = np.random.default_rng(0)
    opens = 10 * np.cumprod(1 + rng.uniform(0.02, 0.09, 400))  # equity compounds to >1e10
    bars = make_bars(opens)
    weights = np.where(np.arange(400) % 2 == 0, 1.0, 0.0)  # all-in / all-out every day
    res = simulate(
        bars, weights, execution=ExecutionConfig(fee_bps=3, slippage_bps=7, min_rebalance_weight=0)
    )
    assert res.final_equity > 1e7
    assert (res.equity["cash"] >= 0).all()
    assert len(res.trades) > 300
