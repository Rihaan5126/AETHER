"""Research-grade robustness analyses (bootstrap intervals, null distributions, costs)."""

from trading_engine.research.robustness import (
    Interval,
    bootstrap_performance,
    cost_sensitivity,
    null_percentile,
    random_null_sharpes,
    stationary_bootstrap_indices,
)

__all__ = [
    "Interval",
    "bootstrap_performance",
    "cost_sensitivity",
    "null_percentile",
    "random_null_sharpes",
    "stationary_bootstrap_indices",
]
