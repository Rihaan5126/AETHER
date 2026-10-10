from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from aether_api.schemas.common import APIModel, UTCDateTime


class AgentSummary(APIModel):
    id: str
    experiment_id: str
    name: str
    algorithm: str
    generation: int
    parent_id: str | None
    origin: str
    seed: int
    is_benchmark: bool
    genome: dict[str, Any]
    has_artifact: bool
    created_at: UTCDateTime
    # latest validation result (if evaluated)
    status: str | None = None
    rank: int | None = None
    score: float | None = None
    total_return: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    warnings: list[dict[str, Any]] = []


class AgentDetail(AgentSummary):
    artifact_path: str | None
    artifact_sha256: str | None
    training_runs: list[dict[str, Any]]
    lineage: list[dict[str, Any]]
    children: list[str]


class LineageNode(BaseModel):
    agent_id: str
    algorithm: str
    generation: int
    origin: str
    is_benchmark: bool
    score: float | None
    rank: int | None
    status: str | None


class LineageEdge(BaseModel):
    parent_id: str
    child_id: str
    operation: str
    mutations: list[str]


class LineageGraph(BaseModel):
    nodes: list[LineageNode]
    edges: list[LineageEdge]


class AgentMetrics(BaseModel):
    agent_id: str
    results: list[dict[str, Any]]
    final_test: list[dict[str, Any]]
