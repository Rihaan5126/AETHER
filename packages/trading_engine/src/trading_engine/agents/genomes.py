"""Typed, serialisable genomes for each agent family.

A genome is the complete, validated configuration of an agent. Each family has
its own Pydantic model; numeric genes declare their bounds with ``ge``/``le``
and mutation metadata in ``json_schema_extra`` so that the evolution engine can
mutate them generically *within* bounds without an untyped parameter bag.

``Genome`` is a discriminated union on ``family``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from trading_engine.features import feature_names

AgentFamily = Literal["random", "buy_and_hold", "momentum", "random_forest", "xgboost", "mlp"]
BENCHMARK_FAMILIES: frozenset[str] = frozenset({"random", "buy_and_hold"})
LEARNED_FAMILIES: frozenset[str] = frozenset({"random_forest", "xgboost", "mlp"})

DEFAULT_FEATURES: tuple[str, ...] = (
    "ret_1", "ret_5", "ret_20", "sma_10_ratio", "sma_50_ratio", "ema_12_ratio", "vol_20",
    "atr_14_pct", "rsi_14", "macd_hist", "mom_60", "drawdown_60", "rel_volume_20",
    "bench_vol_60", "trend_strength",
)


def gene(default: Any, *, ge: float, le: float, log: bool = False, **kw: Any) -> Any:
    """A mutable numeric gene with inclusive bounds."""
    return Field(default, ge=ge, le=le, json_schema_extra={"mutable": True, "log": log}, **kw)


class _GenomeBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PositionPolicy(_GenomeBase):
    """How a forecast becomes a target weight (shared by learned agents)."""

    threshold: float = gene(0.52, ge=0.40, le=0.70)
    sizing: Literal["binary", "proportional"] = "binary"
    max_allocation: float = gene(1.0, ge=0.25, le=1.0)


class RandomGenome(_GenomeBase):
    family: Literal["random"] = "random"
    p_long: float = gene(0.5, ge=0.2, le=0.8)
    switch_prob: float = gene(0.1, ge=0.01, le=0.5)
    max_allocation: float = gene(1.0, ge=0.25, le=1.0)


class BuyAndHoldGenome(_GenomeBase):
    family: Literal["buy_and_hold"] = "buy_and_hold"
    allocation: float = Field(1.0, ge=0.0, le=1.0)


class MomentumGenome(_GenomeBase):
    family: Literal["momentum"] = "momentum"
    fast_window: int = gene(20, ge=3, le=60)
    slow_window: int = gene(100, ge=20, le=250)
    trend_lookback: int = gene(60, ge=10, le=250)
    rule: Literal["ma_cross", "time_series_momentum", "both"] = "ma_cross"
    max_allocation: float = gene(1.0, ge=0.25, le=1.0)

    @model_validator(mode="after")
    def _windows(self) -> MomentumGenome:
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        return self


class _LearnedGenome(_GenomeBase):
    features: tuple[str, ...] = DEFAULT_FEATURES
    policy: PositionPolicy = PositionPolicy()
    train_window: int | None = Field(
        None, ge=250, le=5000,
        description="use only the most recent N training rows (None = all)",
    )

    @field_validator("features")
    @classmethod
    def _known_features(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        known = set(feature_names())
        unknown = [f for f in value if f not in known]
        if unknown:
            raise ValueError(f"unknown features: {unknown}")
        if len(set(value)) != len(value):
            raise ValueError("duplicate features")
        if len(value) < 3:
            raise ValueError("at least 3 features are required")
        order = {name: i for i, name in enumerate(feature_names())}
        return tuple(sorted(value, key=order.__getitem__))


class RandomForestGenome(_LearnedGenome):
    family: Literal["random_forest"] = "random_forest"
    n_estimators: int = gene(200, ge=50, le=500)
    max_depth: int = gene(5, ge=2, le=14)
    min_samples_leaf: int = gene(40, ge=5, le=300)
    max_features: float = gene(0.5, ge=0.1, le=1.0)


class XGBoostGenome(_LearnedGenome):
    family: Literal["xgboost"] = "xgboost"
    n_estimators: int = gene(200, ge=50, le=600)
    max_depth: int = gene(3, ge=2, le=8)
    learning_rate: float = gene(0.05, ge=0.005, le=0.3, log=True)
    subsample: float = gene(0.8, ge=0.5, le=1.0)
    colsample_bytree: float = gene(0.8, ge=0.3, le=1.0)
    reg_lambda: float = gene(1.0, ge=0.01, le=50.0, log=True)
    min_child_weight: float = gene(5.0, ge=1.0, le=50.0, log=True)


class MLPGenome(_LearnedGenome):
    family: Literal["mlp"] = "mlp"
    hidden_layers: tuple[int, ...] = (32, 16)
    dropout: float = gene(0.1, ge=0.0, le=0.5)
    learning_rate: float = gene(1e-3, ge=1e-4, le=1e-2, log=True)
    weight_decay: float = gene(1e-4, ge=1e-6, le=1e-2, log=True)
    max_epochs: int = gene(60, ge=5, le=300)
    batch_size: Literal[64, 128, 256] = 128
    patience: int = gene(8, ge=2, le=40)

    @field_validator("hidden_layers")
    @classmethod
    def _layers(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not 1 <= len(value) <= 3:
            raise ValueError("1 to 3 hidden layers")
        if any(w < 4 or w > 256 for w in value):
            raise ValueError("hidden widths must be within [4, 256]")
        return value


Genome = Annotated[
    RandomGenome | BuyAndHoldGenome | MomentumGenome | RandomForestGenome | XGBoostGenome
    | MLPGenome,
    Field(discriminator="family"),
]
GENOME_ADAPTER: TypeAdapter[Genome] = TypeAdapter(Genome)

GENOME_TYPES: dict[str, type[BaseModel]] = {
    "random": RandomGenome,
    "buy_and_hold": BuyAndHoldGenome,
    "momentum": MomentumGenome,
    "random_forest": RandomForestGenome,
    "xgboost": XGBoostGenome,
    "mlp": MLPGenome,
}


def parse_genome(data: dict[str, Any] | BaseModel) -> Genome:
    if isinstance(data, BaseModel):
        data = data.model_dump(mode="json")
    return GENOME_ADAPTER.validate_python(data)


def default_genome(family: str) -> Genome:
    return parse_genome({"family": family})


def genome_to_dict(genome: BaseModel) -> dict[str, Any]:
    return genome.model_dump(mode="json")
