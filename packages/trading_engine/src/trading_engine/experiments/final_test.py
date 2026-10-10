"""Locked final-test evaluation.

Run once, after evolution has finished, on agents that were *already selected*
using development data only. The agents passed in must already be fitted on the
development period (the API loads them from their verified artifacts). Each one
is simulated once on the untouched test period with the experiment's execution
and risk settings, next to a buy-and-hold benchmark.

The caller (the API service) records every run and refuses a second run unless
the user explicitly acknowledges that the test set then becomes part of
development history ("test contamination").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from trading_engine.agents import BaseTradingAgent, Observations, create_agent
from trading_engine.agents.genomes import BuyAndHoldGenome
from trading_engine.evaluation import compute_metrics
from trading_engine.simulation import simulate
from trading_engine.tournaments import EvaluationConfig
from trading_engine.training import DevelopmentData, TestData

BENCHMARK_ID = "__benchmark__"


@dataclass
class FinalTestResult:
    test_start: str
    test_end: str
    benchmark_metrics: dict[str, Any]
    agents: list[dict[str, Any]] = field(default_factory=list)
    equity: dict[str, pd.DataFrame] = field(default_factory=dict)


def run_final_test(
    agents: Sequence[BaseTradingAgent],
    dev: DevelopmentData,
    test: TestData,
    cfg: EvaluationConfig,
) -> FinalTestResult:
    if not isinstance(test, TestData) or len(test.index) < 20:
        raise ValueError("final test period must contain at least 20 sessions")
    if any(not a.fitted for a in agents):
        raise ValueError("final-test agents must be fitted on development data first")
    history = pd.concat([dev.bars, test.bars])
    obs = Observations(test.symbol, test.features, history)

    def run(agent: BaseTradingAgent) -> Any:
        return simulate(
            test.bars,
            agent.predict(obs),
            execution=cfg.execution,
            risk=cfg.risk,
            realised_vol=test.features["vol_20"],
        )

    bench = run(create_agent(BENCHMARK_ID, BuyAndHoldGenome(), 0))
    bench_eq = bench.equity["equity"]
    result = FinalTestResult(
        test.start.date().isoformat(),
        test.end.date().isoformat(),
        compute_metrics(bench, bench_eq).to_dict(),
    )
    result.equity[BENCHMARK_ID] = bench.equity
    for agent in agents:
        res = run(agent)
        m = compute_metrics(res, bench_eq)
        result.agents.append(
            {
                "agent_id": agent.agent_id,
                "family": agent.family,
                "metrics": m.to_dict(),
                "rejected_orders": len(res.rejected_orders),
            }
        )
        result.equity[agent.agent_id] = res.equity
    return result
