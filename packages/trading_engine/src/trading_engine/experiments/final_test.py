"""Locked final-test evaluation.

Run once, after evolution has finished, on agents that were *already selected*
using development data. Each agent is refitted on the full development period
and simulated once on the untouched test period with the experiment's execution
and risk settings. The caller (the API service) records the run and refuses a
second run unless the user explicitly acknowledges that the test set becomes part
of development history ("test contamination").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from trading_engine.agents import Observations, create_agent
from trading_engine.agents.genomes import BuyAndHoldGenome
from trading_engine.evaluation import compute_metrics
from trading_engine.experiments.runner import fit_on_development
from trading_engine.tournaments import AgentSpec, EvaluationConfig
from trading_engine.training import DevelopmentData, TestData


@dataclass
class FinalTestResult:
    test_start: str
    test_end: str
    benchmark_metrics: dict[str, Any]
    agents: list[dict[str, Any]] = field(default_factory=list)
    equity: dict[str, pd.DataFrame] = field(default_factory=dict)


def run_final_test(
    specs: Sequence[AgentSpec],
    dev: DevelopmentData,
    test: TestData,
    cfg: EvaluationConfig,
) -> FinalTestResult:
    if not isinstance(test, TestData) or len(test.index) < 20:
        raise ValueError("final test period must contain at least 20 sessions")
    history = pd.concat([dev.bars, test.bars])
    obs = Observations(test.symbol, test.features, history)

    def run(agent_spec: AgentSpec) -> tuple[dict[str, Any], pd.DataFrame]:
        if agent_spec.genome.family == "buy_and_hold":  # type: ignore[attr-defined]
            agent = create_agent(agent_spec.agent_id, agent_spec.genome, agent_spec.seed)
        else:
            agent = fit_on_development(agent_spec, dev, horizon=cfg.walk_forward.horizon,
                                       max_seconds=cfg.max_fit_seconds)
        from trading_engine.simulation import simulate  # noqa: PLC0415

        res = simulate(test.bars, agent.predict(obs), execution=cfg.execution, risk=cfg.risk,
                       realised_vol=test.features["vol_20"])
        return {"result": res}, res.equity

    bench_spec = AgentSpec("benchmark", BuyAndHoldGenome(), 0)
    bench_out, bench_eq = run(bench_spec)
    bench_metrics = compute_metrics(bench_out["result"], bench_eq["equity"]).to_dict()
    result = FinalTestResult(test.start.date().isoformat(), test.end.date().isoformat(),
                             bench_metrics)
    for spec in specs:
        out, eq = run(spec)
        m = compute_metrics(out["result"], bench_eq["equity"])
        result.agents.append({"agent_id": spec.agent_id, "family": spec.family,
                              "metrics": m.to_dict(), "n_trades": m.n_trades})
        result.equity[spec.agent_id] = eq
    return result
