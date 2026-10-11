"""Phase 2 research study: does evolution find anything that generalises?

Design (pre-registered: written to the output before any model is trained)
---------------------------------------------------------------------------
Data: one immutable Yahoo dataset (daily, split/dividend adjusted) holding
SPY, QQQ, AAPL, MSFT and NVDA. For each symbol::

    |<----------------- outer development (80%) ----------------->|<- final test (20%) ->|
    |<------ selection data (80% of dev) ------>|<- inner holdout ->|

* Evolution, ranking and the random-search control see **selection data only**
  (walk-forward folds inside it).
* The **inner holdout** is used once per run to ask whether the agents that
  selection preferred also do better on later, unseen data. It never feeds
  back into selection.
* The **outer final test** is touched once, at the very end, only for the
  pre-registered champions, after refitting them on the full development
  period. Nothing is changed in response to it.

Selection rules (fixed in advance):
* initial champion  = rank-1 eligible agent of generation 0
* evolved champion  = rank-1 eligible agent of the last generation
* search champion   = rank-1 eligible agent of a random-search control that
  evaluates the same generation-0 agents plus random genomes, with the same
  number of unique evaluations as the evolution run used.

Usage::

    uv run python scripts/research/phase2_study.py --phase selection
    uv run python scripts/research/phase2_study.py --phase final

Outputs go to docs/research/: ``phase2_preregistration.json`` and
``phase2_results.json``. The selection phase checkpoints after every run and
resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from aether_api.core.config import REPO_ROOT
from aether_api.database.models import Dataset
from aether_api.database.session import session_scope
from aether_api.services.storage import dataset_store
from sqlalchemy import select
from trading_engine.agents.genomes import BENCHMARK_FAMILIES, behaviour_key, genome_to_dict
from trading_engine.evolution import EvolutionConfig, random_genome, run_evolution
from trading_engine.experiments import ExperimentConfig, fit_on_development, prepare_market_data
from trading_engine.experiments.final_test import BENCHMARK_ID, run_final_test
from trading_engine.experiments.manifest import code_version, dependency_versions
from trading_engine.research import (
    bootstrap_performance,
    cost_sensitivity,
    null_percentile,
    random_null_sharpes,
)
from trading_engine.tournaments import (
    AgentEvaluation,
    AgentSpec,
    EvaluationConfig,
    rank_evaluations,
    run_tournament,
)
from trading_engine.training import MarketFrame, SplitConfig, split_development_test

OUT = REPO_ROOT / "docs" / "research"
SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
MAIN = {"population": 8, "generations": 3, "seeds": [1, 2, 3]}
SCALING = {"symbol": "SPY", "configs": [(16, 5), (32, 5)], "seeds": [1, 2]}
OUTER_TEST_FRACTION = 0.2
INNER_HOLDOUT_FRACTION = 0.2
NULL_STRATEGIES = 300
BOOT = {"n_boot": 2000, "mean_block": 20}
EVAL = EvaluationConfig()  # 1 bp fee + 5 bp slippage, 3 expanding folds, 5-bar embargo


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def daily_returns(equity: pd.Series, initial: float) -> pd.Series:
    r = equity.pct_change()
    r.iloc[0] = equity.iloc[0] / initial - 1
    return r


def resolve_dataset(dataset_id: str) -> str:
    with session_scope() as db:
        ds = db.scalars(select(Dataset).where(Dataset.id == dataset_id)).one()
        return ds.hash


# ---- one evaluation window (inner holdout or outer test) ---------------------------------
def evaluate_on_window(
    specs: list[AgentSpec],
    fit_on: Any,
    window: Any,
    *,
    seed: int,
    with_costs_for: str | None = None,
) -> dict[str, Any]:
    """Fit each spec on ``fit_on`` (all usable rows) and simulate once on ``window``."""
    agents = [
        fit_on_development(
            s, fit_on, horizon=EVAL.walk_forward.horizon, max_seconds=EVAL.max_fit_seconds
        )
        for s in specs
    ]
    result = run_final_test(agents, fit_on, window, EVAL)
    initial = EVAL.execution.initial_capital
    bench_r = daily_returns(result.equity[BENCHMARK_ID]["equity"], initial)
    null = random_null_sharpes(
        window.bars,
        window.features,
        n=NULL_STRATEGIES,
        execution=EVAL.execution,
        symbol=window.symbol,
        seed=seed,
    )
    finite = null[np.isfinite(null)]
    out: dict[str, Any] = {
        "start": result.test_start,
        "end": result.test_end,
        "sessions": len(window.index),
        "buy_and_hold": {
            **{
                k: result.benchmark_metrics[k]
                for k in ("total_return", "sharpe_ratio", "max_drawdown")
            },
            "bootstrap": bootstrap_performance(bench_r, seed=seed, **BOOT),
        },
        "random_null": {
            "n": len(finite),
            "sharpe_p5": float(np.quantile(finite, 0.05)),
            "sharpe_median": float(np.median(finite)),
            "sharpe_p95": float(np.quantile(finite, 0.95)),
        },
        "agents": {},
    }
    for agent, entry in zip(agents, result.agents, strict=True):
        m = entry["metrics"]
        r = daily_returns(result.equity[agent.agent_id]["equity"], initial)
        row = {
            "family": agent.family,
            **{
                k: m[k]
                for k in (
                    "total_return",
                    "sharpe_ratio",
                    "max_drawdown",
                    "excess_return",
                    "n_round_trips",
                    "exposure",
                    "turnover",
                    "cost_drag",
                )
            },
            "null_percentile": null_percentile(m["sharpe_ratio"], null),
            "bootstrap": bootstrap_performance(r, bench_r, seed=seed, **BOOT),
        }
        if agent.agent_id == with_costs_for:
            from trading_engine.agents import Observations  # noqa: PLC0415

            hist = pd.concat([fit_on.bars, window.bars])
            weights = agent.predict(Observations(window.symbol, window.features, hist))
            row["cost_sensitivity"] = cost_sensitivity(
                window.bars,
                weights,
                EVAL.execution,
                multipliers=(0.0, 1.0, 2.0, 5.0, 10.0),
                benchmark_equity=result.equity[BENCHMARK_ID]["equity"],
            )
        out["agents"][agent.agent_id] = row
    return out


# ---- selection phase ---------------------------------------------------------------------
def summarise_generation(gen: Any) -> dict[str, Any]:
    evs = gen.outcome.evaluations
    active = [e for e in evs if not e.is_benchmark and e.score is not None]
    eligible = [e for e in evs if e.status == "eligible"]
    by_fam = {e.family: e for e in evs if e.is_benchmark}
    return {
        "generation": gen.generation,
        "best_score": max((e.score for e in active), default=None),
        "median_score": float(np.median([e.score for e in active])) if active else None,
        "n_eligible": len(eligible),
        "best_eligible": _ev_summary(eligible[0]) if eligible else None,
        "buy_and_hold_score": by_fam["buy_and_hold"].score if "buy_and_hold" in by_fam else None,
        "random_score": by_fam["random"].score if "random" in by_fam else None,
        "operations": _ops(gen),
    }


def _ops(gen: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for rec in gen.lineage:
        out[rec.operation] = out.get(rec.operation, 0) + 1
    return out


def _ev_summary(e: AgentEvaluation) -> dict[str, Any]:
    m = e.metrics
    return {
        "agent_id": e.agent_id,
        "family": e.family,
        "score": e.score,
        "validation_sharpe": m.get("sharpe_ratio"),
        "train_sharpe_mean": m.get("train_sharpe_mean"),
        "validation_return": m.get("total_return"),
        "validation_excess": m.get("excess_return"),
        "n_round_trips": m.get("n_round_trips"),
        "warnings": [w["code"] for w in e.warnings],
        "genome": e.genome,
        "seed": e.seed,
    }


def random_search(
    sel_dev: Any, evo: Any, budget: int, seed: int, families: tuple[str, ...], workers: int
) -> tuple[list[AgentEvaluation], dict[str, AgentSpec]]:
    gen0 = evo.generations[0]
    base = [e for e in gen0.outcome.evaluations if not e.is_benchmark]
    specs0 = {s.agent_id: s for s in gen0.specs}
    seen = {behaviour_key(specs0[e.agent_id].genome, specs0[e.agent_id].seed) for e in base}
    rng = np.random.default_rng([seed, 4242])
    extra: list[AgentSpec] = []
    i = 0
    while len(base) + len(extra) < budget:
        fam = families[i % len(families)]
        i += 1
        g = random_genome(fam, rng)
        s = int(rng.integers(0, 2**31 - 1))
        key = behaviour_key(g, s)
        if key in seen:
            continue
        seen.add(key)
        extra.append(AgentSpec(f"rs-{len(extra):02d}-{fam}", g, s))
    outcome = run_tournament(extra, sel_dev, EVAL, max_workers=workers) if extra else None
    pool = [replace(e) for e in base] + (
        [replace(e) for e in outcome.evaluations] if outcome else []
    )
    ranked = rank_evaluations(pool, EVAL.ranking)
    specs = {**specs0, **{s.agent_id: s for s in extra}}
    return ranked, specs


def run_selection(
    dataset_hash: str, symbol: str, pop: int, gens: int, seed: int, workers: int
) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = ExperimentConfig(
        name=f"phase2 {symbol}",
        dataset_hash=dataset_hash,
        symbol=symbol,
        seed=seed,
        split=SplitConfig(test_fraction=OUTER_TEST_FRACTION),
    )
    dev, test = prepare_market_data(dataset_store(), cfg)
    sel_dev, holdout = split_development_test(
        MarketFrame(symbol, dev.bars, dev.features),
        SplitConfig(test_fraction=INNER_HOLDOUT_FRACTION),
    )
    evo_cfg = EvolutionConfig(population_size=pop, generations=gens)
    evo = run_evolution(sel_dev, evo_cfg, EVAL, seed=seed, max_workers=workers)
    unique = {
        behaviour_key(s.genome, s.seed)
        for g in evo.generations
        for s in g.specs
        if s.family not in BENCHMARK_FAMILIES
    }
    spec_by_id = {s.agent_id: s for g in evo.generations for s in g.specs}
    gen_summaries = [summarise_generation(g) for g in evo.generations]
    first_elig = [e for e in evo.generations[0].outcome.evaluations if e.status == "eligible"]
    last_elig = [e for e in evo.generations[-1].outcome.evaluations if e.status == "eligible"]

    rs_ranked, rs_specs = random_search(sel_dev, evo, len(unique), seed, evo_cfg.families, workers)
    rs_elig = [e for e in rs_ranked if e.status == "eligible"]

    champions = {
        "initial": first_elig[0].agent_id if first_elig else None,
        "evolved": last_elig[0].agent_id if last_elig else None,
        "search": rs_elig[0].agent_id if rs_elig else None,
    }
    # Population-level holdout check: every distinct non-benchmark agent of gen 0 and of
    # the last generation, plus the search champion.
    holdout_specs: dict[str, AgentSpec] = {}
    for gen in (evo.generations[0], evo.generations[-1]):
        for s in gen.specs:
            if s.family not in BENCHMARK_FAMILIES:
                holdout_specs.setdefault(behaviour_key(s.genome, s.seed), s)
    if champions["search"]:
        s = rs_specs[champions["search"]]
        holdout_specs.setdefault(behaviour_key(s.genome, s.seed), s)
    specs = list(holdout_specs.values())
    # Map champions (possibly behavioural duplicates) onto the evaluated spec ids.
    key_to_id = {k: s.agent_id for k, s in holdout_specs.items()}
    all_specs = {**spec_by_id, **rs_specs}

    def evaluated_id(aid: str | None) -> str | None:
        if aid is None:
            return None
        s = all_specs[aid]
        return key_to_id[behaviour_key(s.genome, s.seed)]

    champ_eval_ids = {k: evaluated_id(v) for k, v in champions.items()}
    hold = evaluate_on_window(
        specs, sel_dev, holdout, seed=seed, with_costs_for=champ_eval_ids["evolved"]
    )
    gen0_ids = {
        evaluated_id(s.agent_id)
        for s in evo.generations[0].specs
        if s.family not in BENCHMARK_FAMILIES
    }
    last_ids = {
        evaluated_id(s.agent_id)
        for s in evo.generations[-1].specs
        if s.family not in BENCHMARK_FAMILIES
    }

    def pop_stats(ids: set[str | None]) -> dict[str, Any]:
        # Sorted: set order varies between processes (hash randomisation), and float
        # addition is not associative, so an unsorted mean can differ in the last bit.
        vals = [
            hold["agents"][i]["sharpe_ratio"]
            for i in sorted(i for i in ids if i)
            if hold["agents"][i]["sharpe_ratio"] is not None
        ]
        return {
            "n": len(vals),
            "mean_sharpe": float(np.mean(vals)) if vals else None,
            "median_sharpe": float(np.median(vals)) if vals else None,
        }

    return {
        "symbol": symbol,
        "population": pop,
        "generations": gens,
        "seed": seed,
        "selection_period": [sel_dev.start.date().isoformat(), sel_dev.end.date().isoformat()],
        "holdout_period": [holdout.start.date().isoformat(), holdout.end.date().isoformat()],
        "outer_test_period": [test.start.date().isoformat(), test.end.date().isoformat()],
        "unique_evaluations": len(unique),
        "generations_summary": gen_summaries,
        "search_best": _ev_summary(rs_elig[0]) if rs_elig else None,
        "champions": champions,
        "champion_specs": {
            k: (
                {
                    "genome": genome_to_dict(all_specs[v].genome),
                    "seed": all_specs[v].seed,
                    "family": all_specs[v].family,
                }
                if v
                else None
            )
            for k, v in champions.items()
        },
        "champion_holdout_ids": champ_eval_ids,
        "holdout": hold,
        "holdout_population": {
            "generation_0": pop_stats(gen0_ids),
            "last_generation": pop_stats(last_ids),
        },
        "wall_seconds": round(time.perf_counter() - started, 1),
    }


# ---- final phase -------------------------------------------------------------------------
def run_final(dataset_hash: str, run: dict[str, Any]) -> dict[str, Any]:
    symbol, seed = run["symbol"], run["seed"]
    cfg = ExperimentConfig(
        name=f"phase2 final {symbol}",
        dataset_hash=dataset_hash,
        symbol=symbol,
        seed=seed,
        split=SplitConfig(test_fraction=OUTER_TEST_FRACTION),
    )
    dev, test = prepare_market_data(dataset_store(), cfg)
    from trading_engine.agents import parse_genome  # noqa: PLC0415

    specs, roles = [], {}
    for role, spec in run["champion_specs"].items():
        if spec is None:
            continue
        key = behaviour_key(parse_genome(spec["genome"]), spec["seed"])
        if key in roles:
            roles[key].append(role)
            continue
        roles[key] = [role]
        specs.append(
            AgentSpec(f"{role}-{spec['family']}", parse_genome(spec["genome"]), spec["seed"])
        )
    evolved = next((s.agent_id for s in specs if s.agent_id.startswith("evolved")), None)
    res = evaluate_on_window(specs, dev, test, seed=seed, with_costs_for=evolved)
    res["roles"] = {s.agent_id: roles[behaviour_key(s.genome, s.seed)] for s in specs}
    return res


# ---- driver ------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["selection", "final"], required=True)
    ap.add_argument("--dataset", default="ds_aa0bb65d9164")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results_path = OUT / "phase2_results.json"
    prereg_path = OUT / "phase2_preregistration.json"
    dataset_hash = resolve_dataset(args.dataset)
    results: dict[str, Any] = json.loads(results_path.read_text()) if results_path.exists() else {}

    if args.phase == "selection":
        if not prereg_path.exists():
            prereg_path.write_text(
                json.dumps(
                    {
                        "written_at": datetime.now(UTC).isoformat(),
                        "dataset_id": args.dataset,
                        "dataset_hash": dataset_hash,
                        "code_version": code_version(),
                        "dependencies": dependency_versions(),
                        "symbols": SYMBOLS,
                        "main_grid": MAIN,
                        "scaling": SCALING,
                        "outer_test_fraction": OUTER_TEST_FRACTION,
                        "inner_holdout_fraction": INNER_HOLDOUT_FRACTION,
                        "evaluation_config": EVAL.model_dump(mode="json"),
                        "bootstrap": BOOT,
                        "null_strategies": NULL_STRATEGIES,
                        "selection_rules": {
                            "initial": "rank-1 eligible agent of generation 0",
                            "evolved": "rank-1 eligible agent of the last generation",
                            "search": (
                                "rank-1 eligible agent of the equal-budget random-search control"
                            ),
                        },
                        "final_test_policy": (
                            "main-grid champions only; refit on full development; run once; "
                            "no changes afterwards"
                        ),
                    },
                    indent=2,
                )
            )
        runs = [
            (s, MAIN["population"], MAIN["generations"], seed)
            for s in SYMBOLS
            for seed in MAIN["seeds"]
        ]
        runs += [
            (SCALING["symbol"], p, g, seed)
            for p, g in SCALING["configs"]
            for seed in SCALING["seeds"]
        ]
        sel = results.setdefault("selection", {})
        for symbol, pop, gens, seed in runs:
            key = f"{symbol}|{pop}x{gens}|seed{seed}"
            if key in sel:
                continue
            log(f"selection {key}")
            sel[key] = run_selection(dataset_hash, symbol, pop, gens, seed, args.workers)
            results_path.write_text(json.dumps(results, indent=1, default=float))
            log(f"done {key} in {sel[key]['wall_seconds']}s")
    else:
        if "final" in results:
            raise SystemExit("the final test has already been run; refusing to run it again")
        sel = results.get("selection", {})
        main_keys = [
            f"{s}|{MAIN['population']}x{MAIN['generations']}|seed{seed}"
            for s in SYMBOLS
            for seed in MAIN["seeds"]
        ]
        missing = [k for k in main_keys if k not in sel]
        if missing:
            raise SystemExit(f"selection incomplete: {missing}")
        final: dict[str, Any] = {
            "run_at": datetime.now(UTC).isoformat(),
            "code_version": code_version(),
            "runs": {},
        }
        for k in main_keys:
            log(f"final test {k}")
            final["runs"][k] = run_final(dataset_hash, sel[k])
        results["final"] = final
        results_path.write_text(json.dumps(results, indent=1, default=float))
    log("finished")


if __name__ == "__main__":
    main()
