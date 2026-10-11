from __future__ import annotations

from dataclasses import replace

from trading_engine.evaluation import statistical_warnings
from trading_engine.evaluation.metrics import PerformanceMetrics

BASE = PerformanceMetrics(
    sessions=504,
    total_return=0.2,
    annualized_return=0.095,
    annualized_volatility=0.15,
    sharpe_ratio=0.6,
    sortino_ratio=0.8,
    max_drawdown=0.12,
    calmar_ratio=0.8,
    win_rate=0.55,
    profit_factor=1.3,
    n_trades=80,
    n_round_trips=40,
    turnover=4.0,
    exposure=0.6,
    avg_holding_period=6.0,
    total_fees=20.0,
    total_slippage=30.0,
    cost_drag=0.005,
    benchmark_return=0.15,
    excess_return=0.05,
    rejected_orders=0,
    risk_events=0,
)


def codes(m: PerformanceMetrics, **kw) -> set[str]:
    return {w["code"] for w in statistical_warnings(m, [m, m], is_active=True, **kw)}


def test_ordinary_result_has_no_warnings():
    assert codes(BASE, train_sharpe=0.9) == set()


def test_strong_sharpe_on_thin_evidence_is_flagged():
    strong = replace(BASE, sharpe_ratio=2.6, n_round_trips=12)
    assert "suspiciously_strong" in codes(strong)
    # Same Sharpe backed by many trades over a long window is not flagged as suspicious.
    long_and_busy = replace(strong, n_round_trips=150, sessions=252 * 6)
    assert "suspiciously_strong" not in codes(long_and_busy)
    # Passive benchmarks are never "suspicious".
    passive = statistical_warnings(strong, [strong], is_active=False)
    assert "suspiciously_strong" not in {w["code"] for w in passive}


def test_train_validation_gap_is_flagged():
    assert "overfit_gap" in codes(BASE, train_sharpe=2.1)
    assert "overfit_gap" not in codes(BASE, train_sharpe=1.4)
    assert "overfit_gap" not in codes(BASE, train_sharpe=None)
