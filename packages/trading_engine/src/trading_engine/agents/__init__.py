"""Trading agents and the agent factory."""

from __future__ import annotations

from pydantic import BaseModel

from trading_engine.agents.base import (
    AgentDecision,
    BaseTradingAgent,
    FitContext,
    Observations,
    TrainingSet,
)
from trading_engine.agents.baselines import BuyAndHoldAgent, MomentumAgent, RandomAgent
from trading_engine.agents.genomes import (
    BENCHMARK_FAMILIES,
    LEARNED_FAMILIES,
    GENOME_TYPES,
    AgentFamily,
    Genome,
    default_genome,
    genome_to_dict,
    parse_genome,
)
from trading_engine.agents.ml import MLPAgent, RandomForestAgent, XGBoostAgent
from trading_engine.storage import ArtifactRef, ArtifactStore

AGENT_CLASSES: dict[str, type[BaseTradingAgent]] = {
    cls.family: cls
    for cls in (RandomAgent, BuyAndHoldAgent, MomentumAgent, RandomForestAgent, XGBoostAgent,
                MLPAgent)
}


def create_agent(agent_id: str, genome: BaseModel, seed: int) -> BaseTradingAgent:
    family = getattr(genome, "family", None)
    if family not in AGENT_CLASSES:
        raise ValueError(f"unknown agent family {family!r}")
    return AGENT_CLASSES[family](agent_id, genome, seed)  # type: ignore[arg-type]


def save_agent(agent: BaseTradingAgent, store: ArtifactStore, key: str) -> ArtifactRef:
    agent.save(store.workdir(key))
    return store.commit(key)


def load_agent(store: ArtifactStore, ref: ArtifactRef) -> BaseTradingAgent:
    """Load a trusted artifact after verifying all file hashes."""
    import json  # noqa: PLC0415

    directory = store.open_verified(ref)
    meta = json.loads((directory / "meta.json").read_text())
    agent = create_agent(meta["agent_id"], parse_genome(meta["genome"]), meta["seed"])
    agent.load(directory)
    return agent


__all__ = [
    "AGENT_CLASSES",
    "BENCHMARK_FAMILIES",
    "GENOME_TYPES",
    "LEARNED_FAMILIES",
    "AgentDecision",
    "AgentFamily",
    "BaseTradingAgent",
    "BuyAndHoldAgent",
    "FitContext",
    "Genome",
    "MLPAgent",
    "MomentumAgent",
    "Observations",
    "RandomAgent",
    "RandomForestAgent",
    "TrainingSet",
    "XGBoostAgent",
    "create_agent",
    "default_genome",
    "genome_to_dict",
    "load_agent",
    "parse_genome",
    "save_agent",
]
