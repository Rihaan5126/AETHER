"""Mutation, crossover and random sampling of typed genomes.

Numeric genes are discovered from each genome model's field metadata (``ge``/``le``
bounds plus ``json_schema_extra={"mutable": True, "log": ...}``). Mutations are
Gaussian steps in linear or log space scaled to the gene's range and clipped to
its bounds, so a mutated genome is always valid by construction; structural
genes (feature subsets, MLP layers, categorical choices) have dedicated operators.
Every applied change is described in plain text for the lineage record.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from annotated_types import Ge, Le
from pydantic import BaseModel, ValidationError

from trading_engine.agents.genomes import (
    GENOME_TYPES,
    MLPGenome,
    MomentumGenome,
    PositionPolicy,
    _LearnedGenome,
    parse_genome,
)
from trading_engine.features import feature_names

MAX_ATTEMPTS = 25


def numeric_genes(model: type[BaseModel]) -> dict[str, tuple[float, float, bool, bool]]:
    """``name -> (low, high, log_scale, is_int)`` for mutable numeric fields."""
    genes = {}
    for name, info in model.model_fields.items():
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        if not extra.get("mutable"):
            continue
        lo = next(m.ge for m in info.metadata if isinstance(m, Ge))
        hi = next(m.le for m in info.metadata if isinstance(m, Le))
        genes[name] = (float(lo), float(hi), bool(extra.get("log")), info.annotation is int)
    return genes


def _step(value: float, lo: float, hi: float, log: bool, is_int: bool,
          rng: np.random.Generator, strength: float) -> float:
    if log:
        span = math.log(hi) - math.log(lo)
        new = math.exp(math.log(value) + rng.normal(0, strength * span))
    else:
        new = value + rng.normal(0, strength * (hi - lo))
    new = min(max(new, lo), hi)
    if is_int:
        new = int(round(new))
        if new == int(value):
            new = int(min(max(value + (1 if rng.random() < 0.5 else -1), lo), hi))
    return new


def _sample(lo: float, hi: float, log: bool, is_int: bool, rng: np.random.Generator) -> float:
    v = math.exp(rng.uniform(math.log(lo), math.log(hi))) if log else rng.uniform(lo, hi)
    return int(round(v)) if is_int else v


def _fmt(v: Any) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def _mutate_numeric(data: dict[str, Any], model: type[BaseModel], rng: np.random.Generator,
                    strength: float, prefix: str = "") -> list[str] | None:
    genes = numeric_genes(model)
    if not genes:
        return None
    name = list(genes)[int(rng.integers(len(genes)))]
    lo, hi, log, is_int = genes[name]
    old = data[name]
    new = _step(float(old), lo, hi, log, is_int, rng, strength)
    data[name] = new
    return [f"{prefix}{name}: {_fmt(old)} -> {_fmt(new)}"]


def _mutate_features(data: dict[str, Any], rng: np.random.Generator) -> list[str]:
    current = list(data["features"])
    pool = [f for f in feature_names() if f not in current]
    if (rng.random() < 0.5 and pool) or len(current) <= 3:
        add = pool[int(rng.integers(len(pool)))]
        data["features"] = [*current, add]
        return [f"add feature {add}"]
    drop = current[int(rng.integers(len(current)))]
    data["features"] = [f for f in current if f != drop]
    return [f"remove feature {drop}"]


def _mutate_policy(data: dict[str, Any], rng: np.random.Generator, strength: float) -> list[str]:
    policy = dict(data["policy"])
    if rng.random() < 0.15:
        old = policy["sizing"]
        policy["sizing"] = "proportional" if old == "binary" else "binary"
        data["policy"] = policy
        return [f"policy.sizing: {old} -> {policy['sizing']}"]
    out = _mutate_numeric(policy, PositionPolicy, rng, strength, prefix="policy.")
    data["policy"] = policy
    return out or []


def _mutate_layers(data: dict[str, Any], rng: np.random.Generator) -> list[str]:
    layers = list(data["hidden_layers"])
    old = tuple(layers)
    r = rng.random()
    if r < 0.2 and len(layers) < 3:
        layers.append(max(4, layers[-1] // 2))
    elif r < 0.4 and len(layers) > 1:
        layers.pop()
    else:
        i = int(rng.integers(len(layers)))
        layers[i] = int(min(max(round(layers[i] * float(rng.choice([0.5, 0.75, 1.5, 2.0]))), 4),
                            256))
    data["hidden_layers"] = layers
    return [f"hidden_layers: {old} -> {tuple(layers)}"]


def mutate(genome: BaseModel, rng: np.random.Generator, *, strength: float = 0.15,
           n_changes: int = 1) -> tuple[BaseModel, list[str]]:
    """Return a valid mutated copy of ``genome`` and a description of the changes."""
    model = type(genome)
    for _ in range(MAX_ATTEMPTS):
        data = genome.model_dump(mode="json")
        notes: list[str] = []
        for _ in range(n_changes):
            notes += _one_mutation(data, model, rng, strength)
        try:
            child = parse_genome(data)
        except ValidationError:
            continue
        if child != genome:
            return child, notes
    return genome, ["no valid mutation found; genome unchanged"]


def _one_mutation(data: dict[str, Any], model: type[BaseModel], rng: np.random.Generator,
                  strength: float) -> list[str]:
    ops: list[str] = ["numeric"]
    if issubclass(model, _LearnedGenome):
        ops += ["features", "policy", "train_window"]
    if model is MLPGenome:
        ops += ["layers"]
    if model is MomentumGenome:
        ops += ["rule"]
    op = ops[int(rng.integers(len(ops)))]
    if op == "features":
        return _mutate_features(data, rng)
    if op == "policy":
        return _mutate_policy(data, rng, strength)
    if op == "layers":
        return _mutate_layers(data, rng)
    if op == "train_window":
        old = data["train_window"]
        data["train_window"] = None if old is not None else int(rng.choice([500, 750, 1000, 1500]))
        return [f"train_window: {old} -> {data['train_window']}"]
    if op == "rule":
        old = data["rule"]
        choices = [r for r in ("ma_cross", "time_series_momentum", "both") if r != old]
        data["rule"] = choices[int(rng.integers(len(choices)))]
        return [f"rule: {old} -> {data['rule']}"]
    return _mutate_numeric(data, model, rng, strength) or []


def crossover(a: BaseModel, b: BaseModel, rng: np.random.Generator) -> tuple[BaseModel, list[str]]:
    """Uniform crossover for two genomes of the *same* family."""
    if type(a) is not type(b):
        raise ValueError("crossover requires genomes of the same family")
    da, db = a.model_dump(mode="json"), b.model_dump(mode="json")
    for _ in range(MAX_ATTEMPTS):
        child: dict[str, Any] = {}
        taken_b = []
        for key in da:
            if key == "features":
                union = sorted(set(da[key]) | set(db[key]))
                keep = [f for f in union if (f in da[key] and f in db[key]) or rng.random() < 0.5]
                child[key] = keep if len(keep) >= 3 else da[key]
                continue
            if rng.random() < 0.5:
                child[key] = da[key]
            else:
                child[key] = db[key]
                if da[key] != db[key]:
                    taken_b.append(key)
        try:
            return parse_genome(child), [f"crossover; genes from parent B: {taken_b or 'none'}"]
        except ValidationError:
            continue
    return a, ["crossover produced no valid child; copied parent A"]


def random_genome(family: str, rng: np.random.Generator) -> BaseModel:
    """Sample a genome uniformly within its bounds (used for immigrants and diversity)."""
    model = GENOME_TYPES[family]
    for _ in range(MAX_ATTEMPTS):
        data: dict[str, Any] = {"family": family}
        for name, (lo, hi, log, is_int) in numeric_genes(model).items():
            data[name] = _sample(lo, hi, log, is_int, rng)
        if issubclass(model, _LearnedGenome):
            names = feature_names()
            k = int(rng.integers(6, len(names) + 1))
            data["features"] = sorted(rng.choice(names, size=k, replace=False).tolist())
            pol = {n: _sample(lo, hi, log, is_int, rng)
                   for n, (lo, hi, log, is_int) in numeric_genes(PositionPolicy).items()}
            pol["threshold"] = float(rng.uniform(0.48, 0.58))
            pol["sizing"] = "binary" if rng.random() < 0.7 else "proportional"
            data["policy"] = pol
        if model is MLPGenome:
            depth = int(rng.integers(1, 3))
            data["hidden_layers"] = [int(rng.choice([8, 16, 32, 64])) for _ in range(depth)]
            data["max_epochs"] = int(rng.integers(20, 80))
        if model is MomentumGenome:
            data["rule"] = str(rng.choice(["ma_cross", "time_series_momentum", "both"]))
        try:
            return parse_genome(data)
        except ValidationError:
            continue
    return parse_genome({"family": family})
