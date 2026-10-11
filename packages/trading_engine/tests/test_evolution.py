from __future__ import annotations

import numpy as np
import pytest
import trading_engine.tournaments.engine as engine_mod
from trading_engine.agents import create_agent, default_genome, parse_genome
from trading_engine.agents.genomes import GENOME_TYPES, MomentumGenome, behaviour_key
from trading_engine.evolution import (
    EvolutionConfig,
    crossover,
    mutate,
    numeric_genes,
    random_genome,
    run_evolution,
)
from trading_engine.tournaments import AgentSpec, EvaluationConfig, run_tournament
from trading_engine.training import WalkForwardConfig

FAST_EVAL = EvaluationConfig(walk_forward=WalkForwardConfig(n_folds=2))
SMALL = EvolutionConfig(
    population_size=4,
    generations=2,
    elite_count=1,
    immigrant_count=1,
    families=("momentum", "xgboost"),
)


@pytest.mark.parametrize("family", ["momentum", "random_forest", "xgboost", "mlp", "random"])
def test_mutation_respects_bounds(family):
    rng = np.random.default_rng(0)
    genome = default_genome(family)
    bounds = numeric_genes(GENOME_TYPES[family])
    for _ in range(200):
        genome, notes = mutate(genome, rng, strength=0.5, n_changes=2)
        assert notes
        data = genome.model_dump()
        for name, (lo, hi, _, is_int) in bounds.items():
            assert lo <= data[name] <= hi
            if is_int:
                assert isinstance(data[name], int)
        assert parse_genome(genome.model_dump(mode="json")) == genome


def test_behaviour_key_ignores_unused_genes():
    # ma_cross never reads trend_lookback, so these two agents trade identically.
    a = MomentumGenome(rule="ma_cross", trend_lookback=60)
    b = MomentumGenome(rule="ma_cross", trend_lookback=37)
    assert a != b
    assert behaviour_key(a) == behaviour_key(b)
    c = MomentumGenome(rule="time_series_momentum", fast_window=10, slow_window=50)
    d = MomentumGenome(rule="time_series_momentum", fast_window=20, slow_window=100)
    assert behaviour_key(c) == behaviour_key(d)
    assert behaviour_key(MomentumGenome(rule="both", trend_lookback=37)) != behaviour_key(
        MomentumGenome(rule="both", trend_lookback=60)
    )
    # The seed matters only where it changes the fitted model.
    assert behaviour_key(a, 1) == behaviour_key(a, 2)
    xgb = default_genome("xgboost")
    assert behaviour_key(xgb, 1) != behaviour_key(xgb, 2)


@pytest.mark.parametrize("family", ["momentum", "random_forest", "xgboost", "mlp"])
def test_mutation_always_changes_behaviour(family):
    """Regression: a mutation of an inactive gene used to create a parent clone."""
    rng = np.random.default_rng(3)
    genome = default_genome(family)
    for _ in range(300):
        child, _ = mutate(genome, rng, strength=0.3)
        assert behaviour_key(child) != behaviour_key(genome)
        genome = child


def test_random_genomes_are_valid():
    rng = np.random.default_rng(1)
    for family in ("momentum", "random_forest", "xgboost", "mlp"):
        for _ in range(30):
            g = random_genome(family, rng)
            assert parse_genome(g.model_dump(mode="json")) == g


def test_crossover_requires_same_family():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        crossover(default_genome("xgboost"), default_genome("mlp"), rng)
    child, notes = crossover(default_genome("xgboost"), random_genome("xgboost", rng), rng)
    assert child.family == "xgboost" and notes


@pytest.fixture(scope="module")
def evolution(dev_test):
    dev, _ = dev_test
    return run_evolution(dev, SMALL, FAST_EVAL, seed=7)


def test_generations_and_lineage(evolution):
    assert len(evolution.generations) == 2
    gen1 = evolution.generations[1]
    ids0 = {s.agent_id for s in evolution.generations[0].specs}
    ops = {rec.operation for rec in gen1.lineage}
    assert {"elite", "immigrant", "benchmark"} <= ops
    for rec in gen1.lineage:
        if rec.operation in ("mutation", "crossover", "elite"):
            assert rec.parent_ids and set(rec.parent_ids) <= ids0
            assert rec.genome_before is not None and rec.mutations
        assert rec.validation_score is not None or rec.operation == "benchmark"


def test_elite_is_preserved_with_identical_score(evolution):
    gen0 = {e.agent_id: e for e in evolution.generations[0].outcome.evaluations}
    gen1 = {e.agent_id: e for e in evolution.generations[1].outcome.evaluations}
    elites = [r for r in evolution.generations[1].lineage if r.operation == "elite"]
    assert elites
    best0 = next(e for e in evolution.generations[0].outcome.evaluations if e.status == "eligible")
    assert elites[0].parent_ids == [best0.agent_id]
    assert elites[0].genome_after == best0.genome
    assert gen1[elites[0].child_id].score == pytest.approx(gen0[best0.agent_id].score)


def test_generations_have_no_behavioural_duplicates(dev_test):
    dev, _ = dev_test
    cfg = EvolutionConfig(
        population_size=6, generations=3, elite_count=2, immigrant_count=1, families=("momentum",)
    )
    result = run_evolution(dev, cfg, FAST_EVAL, seed=11)
    for gen in result.generations:
        keys = [behaviour_key(s.genome, s.seed) for s in gen.specs if s.family == "momentum"]
        assert len(keys) == cfg.population_size
        assert len(set(keys)) == len(keys), f"duplicate agents in generation {gen.generation}"


def test_benchmarks_never_reproduce(evolution):
    bench_ids = {
        e.agent_id for g in evolution.generations for e in g.outcome.evaluations if e.is_benchmark
    }
    for rec in evolution.lineage:
        assert not set(rec.parent_ids) & bench_ids


def test_selection_is_reproducible(dev_test, evolution):
    dev, _ = dev_test
    again = run_evolution(dev, SMALL, FAST_EVAL, seed=7)
    a = [(r.child_id, r.operation, r.parent_ids, r.genome_after) for r in evolution.lineage]
    b = [(r.child_id, r.operation, r.parent_ids, r.genome_after) for r in again.lineage]
    assert a == b


def test_final_test_data_never_reaches_agents(dev_test, monkeypatch):
    """Spy on every agent the engine creates during evolution."""
    dev, test = dev_test
    seen: list = []
    real_create = engine_mod.create_agent

    def spying_create(agent_id, genome, seed):
        agent = real_create(agent_id, genome, seed)
        fit, predict = agent.fit, agent.predict

        def fit_spy(training, context):
            seen.append(training.X.index.max())
            seen.append(training.history.index.max())
            return fit(training, context)

        def predict_spy(obs):
            seen.append(obs.features.index.max())
            seen.append(obs.history.index.max())
            return predict(obs)

        agent.fit, agent.predict = fit_spy, predict_spy
        return agent

    monkeypatch.setattr(engine_mod, "create_agent", spying_create)
    run_evolution(dev, SMALL.model_copy(update={"generations": 1}), FAST_EVAL, seed=3)
    assert seen and max(seen) < test.start


def test_tournament_and_evolution_refuse_test_data(dev_test):
    _, test = dev_test
    with pytest.raises(TypeError):
        run_tournament([AgentSpec("x", default_genome("momentum"), 1)], test, FAST_EVAL)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        run_evolution(test, SMALL, FAST_EVAL, seed=1)  # type: ignore[arg-type]


def test_tournament_ranking_policy(dev_test):
    dev, _ = dev_test
    specs = [
        AgentSpec(f"a{i}", default_genome(f), 1)
        for i, f in enumerate(["momentum", "random", "buy_and_hold", "xgboost"])
    ]
    out = run_tournament(specs, dev, FAST_EVAL)
    ranks = [e.rank for e in out.evaluations]
    assert ranks == list(range(1, 5))
    for e in out.evaluations:
        assert e.status in {"eligible", "benchmark", "insufficient_evidence", "excluded"}
        assert e.equity is not None and len(e.equity) > 0
        assert len(e.fold_metrics) == 2
        if e.status != "excluded":
            comps = e.score_components
            total = (
                comps["mean_fold_sharpe"]
                + comps["stability_penalty"]
                + comps["drawdown_penalty"]
                + comps["turnover_penalty"]
            )
            assert e.score == pytest.approx(total)
    assert {e.status for e in out.evaluations if e.is_benchmark} == {"benchmark"}
    bh = next(e for e in out.evaluations if e.family == "buy_and_hold")
    assert bh.metrics["excess_return"] == pytest.approx(0.0, abs=1e-12)


def test_failed_agent_is_recorded_not_dropped(dev_test, monkeypatch):
    dev, _ = dev_test

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    agent = create_agent("x", default_genome("xgboost"), 1)
    monkeypatch.setattr(type(agent), "_fit_model", broken)
    out = run_tournament([AgentSpec("bad", default_genome("xgboost"), 1)], dev, FAST_EVAL)
    ev = out.evaluations[0]
    assert ev.status == "excluded" and "boom" in (ev.error or "")


def test_parallel_tournament_matches_serial(dev_test):
    """Process-pool evaluation must give bit-identical results to the serial path."""
    dev, _ = dev_test
    specs = [
        AgentSpec("p-mom", default_genome("momentum"), 1),
        AgentSpec("p-xgb", parse_genome({"family": "xgboost", "n_estimators": 60}), 2),
        AgentSpec("p-mlp", parse_genome({"family": "mlp", "max_epochs": 8}), 3),
    ]
    serial = run_tournament(specs, dev, FAST_EVAL, max_workers=1)
    parallel = run_tournament(specs, dev, FAST_EVAL, max_workers=2)
    a = {e.agent_id: (e.score, e.metrics, e.fold_metrics) for e in serial.evaluations}
    b = {e.agent_id: (e.score, e.metrics, e.fold_metrics) for e in parallel.evaluations}
    assert a == b


def test_stored_tracebacks_do_not_reveal_host_paths():
    from trading_engine.tournaments.engine import _portable_traceback

    text = (
        "Traceback (most recent call last):\n"
        r'  File "C:\Users\someone\proj\packages\trading_engine\src\trading_engine'
        r'\simulation\simulator.py", line 200, in simulate'
        "\n"
        '  File "/home/someone/.venv/lib/python3.13/site-packages/pandas/core/frame.py", line 1\n'
        '  File "<string>", line 3\n'
    )
    out = _portable_traceback(text)
    assert "someone" not in out
    assert 'File "trading_engine/simulation/simulator.py", line 200' in out
    assert 'File "site-packages/pandas/core/frame.py"' in out
    assert 'File "<string>"' in out
