"""Evolutionary loop over agent genomes.

    initial population -> walk-forward tournament (development data only)
      -> rank -> elites + tournament-selected parents -> mutation / crossover
      -> random immigrants -> next population -> repeat

Scientific constraint: :func:`run_evolution` accepts only
:class:`~trading_engine.training.splits.DevelopmentData`, whose construction
guarantees it contains no bar from the final test period. Test results cannot
reach selection because they are never computed here.

Benchmarks (random and buy-and-hold) are evaluated in every generation for
reference but never reproduce. Evaluations are cached by (genome, seed), so an
elite carried into the next generation keeps exactly its previous score.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from trading_engine.agents.genomes import (
    BuyAndHoldGenome,
    Genome,
    RandomGenome,
    behaviour_key,
    default_genome,
    genome_to_dict,
)
from trading_engine.evolution.operators import crossover, mutate, random_genome
from trading_engine.tournaments import (
    AgentEvaluation,
    AgentSpec,
    EvaluationConfig,
    TournamentOutcome,
    run_tournament,
)
from trading_engine.training.splits import DevelopmentData

EVOLVABLE_FAMILIES = ("momentum", "random_forest", "xgboost", "mlp")


class EvolutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    population_size: int = Field(8, ge=2, le=64)
    generations: int = Field(3, ge=1, le=20)
    elite_count: int = Field(2, ge=0, le=16)
    tournament_size: int = Field(3, ge=2, le=8)
    immigrant_count: int = Field(1, ge=0, le=8)
    crossover_rate: float = Field(0.25, ge=0, le=1)
    mutation_strength: float = Field(0.15, gt=0, le=1)
    families: tuple[str, ...] = EVOLVABLE_FAMILIES
    include_benchmarks: bool = True

    @field_validator("families")
    @classmethod
    def _families(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        bad = [f for f in value if f not in EVOLVABLE_FAMILIES]
        if bad or not value:
            raise ValueError(f"families must be a non-empty subset of {EVOLVABLE_FAMILIES}")
        return value


@dataclass
class LineageRecord:
    child_id: str
    generation: int
    operation: str  # initial | elite | mutation | crossover | immigrant | benchmark
    parent_ids: list[str]
    mutations: list[str]
    genome_before: dict[str, Any] | None
    genome_after: dict[str, Any]
    seed: int
    validation_score: float | None = None


@dataclass
class GenerationResult:
    generation: int
    specs: list[AgentSpec]
    outcome: TournamentOutcome
    lineage: list[LineageRecord]


@dataclass
class EvolutionResult:
    generations: list[GenerationResult] = field(default_factory=list)

    @property
    def lineage(self) -> list[LineageRecord]:
        return [rec for g in self.generations for rec in g.lineage]

    def best(self) -> AgentEvaluation | None:
        last = self.generations[-1].outcome.evaluations if self.generations else []
        eligible = [e for e in last if e.status == "eligible"]
        return eligible[0] if eligible else None


def _genome_key(genome: BaseModel, seed: int) -> str:
    blob = json.dumps({"g": genome_to_dict(genome), "s": seed}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def initial_population(config: EvolutionConfig, rng: np.random.Generator) -> list[Genome]:
    """Default genome for each family first, then random genomes for diversity."""
    pop: list[Genome] = [default_genome(f) for f in config.families][: config.population_size]
    while len(pop) < config.population_size:
        family = config.families[len(pop) % len(config.families)]
        pop.append(random_genome(family, rng))
    return pop


def tournament_select(
    ranked: Sequence[AgentEvaluation], k: int, rng: np.random.Generator
) -> AgentEvaluation:
    """Pick ``k`` random contenders; the best-ranked one wins."""
    idx = rng.choice(len(ranked), size=min(k, len(ranked)), replace=False)
    return min((ranked[int(i)] for i in idx), key=lambda e: e.rank or 10**9)


def run_evolution(
    dev: DevelopmentData,
    config: EvolutionConfig,
    evaluation: EvaluationConfig,
    *,
    seed: int,
    id_factory: Callable[[int, int, str], str] | None = None,
    max_workers: int = 1,
    on_generation: Callable[[GenerationResult], None] | None = None,
    on_agent: Callable[[int, int, int, AgentEvaluation], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> EvolutionResult:
    if not isinstance(dev, DevelopmentData):
        raise TypeError("evolution only accepts DevelopmentData (final test is locked)")
    make_id = id_factory or (lambda g, i, fam: f"g{g}-{i:02d}-{fam}")
    rng = np.random.default_rng(seed)
    cache: dict[str, AgentEvaluation] = {}
    result = EvolutionResult()

    def new_seed() -> int:
        return int(rng.integers(0, 2**31 - 1))

    population: list[tuple[BaseModel, int, LineageRecord]] = []
    for i, genome in enumerate(initial_population(config, rng)):
        s = new_seed()
        cid = make_id(0, i, genome.family)
        population.append(
            (genome, s, LineageRecord(cid, 0, "initial", [], [], None, genome_to_dict(genome), s))
        )

    bench_seed = int(seed)
    for gen in range(config.generations):
        if should_cancel and should_cancel():
            break
        specs = [AgentSpec(rec.child_id, g, s) for g, s, rec in population]
        lineage = [rec for _, _, rec in population]
        if config.include_benchmarks:
            bench_genomes: tuple[Genome, ...] = (RandomGenome(), BuyAndHoldGenome())
            for j, bg in enumerate(bench_genomes):
                bid = make_id(gen, config.population_size + j, bg.family)
                specs.append(AgentSpec(bid, bg, bench_seed))
                lineage.append(
                    LineageRecord(
                        bid, gen, "benchmark", [], [], None, genome_to_dict(bg), bench_seed
                    )
                )
        # Re-use cached evaluations (elites, duplicates); evaluate the rest.
        todo = [s for s in specs if _genome_key(s.genome, s.seed) not in cache]

        def progress(done: int, total: int, ev: AgentEvaluation, _gen: int = gen) -> None:
            if on_agent:
                on_agent(_gen, done, total, ev)

        outcome = run_tournament(
            todo,
            dev,
            evaluation,
            max_workers=max_workers,
            on_progress=progress,
            should_cancel=should_cancel,
        )
        for spec, ev in zip(todo, _by_id(outcome.evaluations, todo), strict=True):
            cache[_genome_key(spec.genome, spec.seed)] = ev
        evaluations = []
        for spec in specs:
            cached = cache[_genome_key(spec.genome, spec.seed)]
            ev = _clone_eval(cached, spec.agent_id)
            evaluations.append(ev)
        from trading_engine.tournaments import rank_evaluations  # noqa: PLC0415

        ranked = rank_evaluations(evaluations, evaluation.ranking)
        outcome.evaluations = ranked
        score_by_id = {e.agent_id: e.score for e in ranked}
        for rec in lineage:
            rec.validation_score = score_by_id.get(rec.child_id)
        gen_result = GenerationResult(gen, specs, outcome, lineage)
        result.generations.append(gen_result)
        if on_generation:
            on_generation(gen_result)
        if gen == config.generations - 1:
            break
        population = _next_population(gen + 1, ranked, specs, config, rng, new_seed, make_id)
    return result


def _by_id(evals: Sequence[AgentEvaluation], specs: Sequence[AgentSpec]) -> list[AgentEvaluation]:
    lookup = {e.agent_id: e for e in evals}
    return [lookup[s.agent_id] for s in specs]


def _clone_eval(ev: AgentEvaluation, agent_id: str) -> AgentEvaluation:
    from dataclasses import replace  # noqa: PLC0415

    cached_from = None if ev.agent_id == agent_id else (ev.cached_from or ev.agent_id)
    return replace(
        ev, agent_id=agent_id, rank=None, score=None, score_components={}, cached_from=cached_from
    )


def _next_population(
    gen: int,
    ranked: Sequence[AgentEvaluation],
    specs: Sequence[AgentSpec],
    config: EvolutionConfig,
    rng: np.random.Generator,
    new_seed: Callable[[], int],
    make_id: Callable[[int, int, str], str],
) -> list[tuple[BaseModel, int, LineageRecord]]:
    spec_by_id = {s.agent_id: s for s in specs}
    candidates = [e for e in ranked if e.status == "eligible"]
    if not candidates:  # nobody passed: fall back to non-benchmark, non-failed agents
        candidates = [e for e in ranked if not e.is_benchmark and e.status != "excluded"]
    if not candidates:
        candidates = [e for e in ranked if not e.is_benchmark]
    out: list[tuple[BaseModel, int, LineageRecord]] = []
    # Behaviourally identical agents waste population slots and fake diversity
    # (e.g. two momentum genomes differing only in an unused gene).
    seen: set[str] = set()

    def add(
        genome: BaseModel,
        s: int,
        op: str,
        parents: list[str],
        notes: list[str],
        before: BaseModel | None,
    ) -> bool:
        key = behaviour_key(genome, s)
        if key in seen:
            return False
        seen.add(key)
        cid = make_id(gen, len(out), genome.family)  # type: ignore[attr-defined]
        out.append(
            (
                genome,
                s,
                LineageRecord(
                    cid,
                    gen,
                    op,
                    parents,
                    notes,
                    genome_to_dict(before) if before else None,
                    genome_to_dict(genome),
                    s,
                ),
            )
        )
        return True

    elites = 0
    for elite in candidates:
        if elites == config.elite_count:
            break
        sp = spec_by_id[elite.agent_id]
        notes = ["carried over unchanged"]
        elites += add(sp.genome, sp.seed, "elite", [elite.agent_id], notes, sp.genome)

    n_immigrants = min(config.immigrant_count, config.population_size - len(out))
    # Bounded retries: a tiny search space could otherwise loop forever.
    attempts, max_attempts = 0, 50 * config.population_size
    while len(out) < config.population_size - n_immigrants and attempts < max_attempts:
        attempts += 1
        parent = tournament_select(candidates, config.tournament_size, rng)
        pspec = spec_by_id[parent.agent_id]
        same_family = [
            c for c in candidates if c.family == parent.family and c.agent_id != parent.agent_id
        ]
        if same_family and rng.random() < config.crossover_rate:
            mate = tournament_select(same_family, config.tournament_size, rng)
            mspec = spec_by_id[mate.agent_id]
            child, notes = crossover(pspec.genome, mspec.genome, rng)
            child, more = mutate(child, rng, strength=config.mutation_strength)
            add(
                child,
                new_seed(),
                "crossover",
                [parent.agent_id, mate.agent_id],
                notes + more,
                pspec.genome,
            )
        else:
            n_changes = 1 if rng.random() < 0.7 else 2
            child, notes = mutate(
                pspec.genome, rng, strength=config.mutation_strength, n_changes=n_changes
            )
            add(child, new_seed(), "mutation", [parent.agent_id], notes, pspec.genome)
    # Immigrants also fill any slots left if mutation kept producing duplicates.
    while len(out) < config.population_size and attempts < 2 * max_attempts:
        attempts += 1
        family = config.families[int(rng.integers(len(config.families)))]
        add(random_genome(family, rng), new_seed(), "immigrant", [], ["random immigrant"], None)
    return out
