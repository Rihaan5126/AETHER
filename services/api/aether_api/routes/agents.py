from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from aether_api.core.errors import not_found
from aether_api.database.models import Agent
from aether_api.database.session import get_db
from aether_api.routes.deps import Pagination, paginate
from aether_api.schemas.agents import AgentDetail, AgentMetrics, AgentSummary, LineageGraph
from aether_api.schemas.common import Page
from aether_api.services import queries

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("", response_model=Page[AgentSummary])
def list_agents(
    experiment_id: str | None = None,
    generation: int | None = Query(None, ge=0),
    algorithm: str | None = None,
    include_benchmarks: bool = True,
    page: Pagination = Depends(),
    db: Session = Depends(get_db),
) -> Page[AgentSummary]:
    q = select(Agent).order_by(Agent.created_at.desc(), Agent.generation.desc(), Agent.id)
    if experiment_id:
        q = q.where(Agent.experiment_id == experiment_id)
    if generation is not None:
        q = q.where(Agent.generation == generation)
    if algorithm:
        q = q.where(Agent.algorithm == algorithm)
    if not include_benchmarks:
        q = q.where(Agent.is_benchmark.is_(False))
    items, total = paginate(db, q, page)
    return Page[AgentSummary](
        items=[AgentSummary.model_validate(s) for s in queries.list_agent_summaries(db, items)],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/{agent_id}", response_model=AgentDetail)
def get_agent(agent_id: str, db: Session = Depends(get_db)) -> AgentDetail:
    return AgentDetail.model_validate(queries.agent_detail(db, agent_id))


@router.get("/{agent_id}/lineage", response_model=LineageGraph)
def agent_lineage(agent_id: str, db: Session = Depends(get_db)) -> LineageGraph:
    """Ancestors and descendants of one agent within its experiment."""
    agent = db.get(Agent, agent_id)
    if agent is None:
        raise not_found("agent", agent_id)
    graph = queries.lineage_graph(db, agent.experiment_id)
    parents: dict[str, set[str]] = {}
    children: dict[str, set[str]] = {}
    for e in graph["edges"]:
        parents.setdefault(e["child_id"], set()).add(e["parent_id"])
        children.setdefault(e["parent_id"], set()).add(e["child_id"])
    keep = {agent_id}
    for links in (parents, children):
        stack = [agent_id]
        while stack:
            for nxt in links.get(stack.pop(), ()):
                if nxt not in keep:
                    keep.add(nxt)
                    stack.append(nxt)
    return LineageGraph(
        nodes=[n for n in graph["nodes"] if n["agent_id"] in keep],
        edges=[e for e in graph["edges"] if e["parent_id"] in keep and e["child_id"] in keep],
    )


@router.get("/{agent_id}/metrics", response_model=AgentMetrics)
def agent_metrics(agent_id: str, db: Session = Depends(get_db)) -> AgentMetrics:
    return AgentMetrics.model_validate(queries.agent_metrics(db, agent_id))
