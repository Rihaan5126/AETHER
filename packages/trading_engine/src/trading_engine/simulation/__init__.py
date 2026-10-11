"""Deterministic event-driven portfolio simulator."""

from trading_engine.simulation.simulator import (
    ExecutionConfig,
    RejectedOrder,
    SimulationResult,
    Trade,
    simulate,
)

__all__ = ["ExecutionConfig", "RejectedOrder", "SimulationResult", "Trade", "simulate"]
