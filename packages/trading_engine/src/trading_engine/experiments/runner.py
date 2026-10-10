"""Glue between stored datasets and the research engine."""

from __future__ import annotations

import pandas as pd

from trading_engine.agents import BaseTradingAgent, FitContext, TrainingSet, create_agent
from trading_engine.data import DatasetStore
from trading_engine.data.schema import symbol_frame
from trading_engine.experiments.config import ExperimentConfig
from trading_engine.features import compute_features, feature_names
from trading_engine.tournaments import AgentSpec
from trading_engine.training import (
    DevelopmentData,
    MarketFrame,
    TestData,
    split_development_test,
)
from trading_engine.training.splits import training_matrix, usable_positions

FEATURE_CACHE_VERSION = "v1"


def load_features(store: DatasetStore, dataset_hash: str, symbol: str,
                  bars: pd.DataFrame) -> pd.DataFrame:
    """Compute features once per (dataset, symbol) and cache them as Parquet.

    The cache is keyed by dataset hash and feature-set version; datasets are
    immutable, so the cache never goes stale.
    """
    key = f"features-{FEATURE_CACHE_VERSION}-{symbol}"
    path = store.cache_path(dataset_hash, key)
    if path.is_file():
        cached = pd.read_parquet(path)
        if list(cached.columns) == feature_names() and cached.index.equals(bars.index):
            return cached
    feats = compute_features(bars)
    feats.to_parquet(path)
    return feats


def prepare_market_data(store: DatasetStore, config: ExperimentConfig
                        ) -> tuple[DevelopmentData, TestData]:
    dataset = store.load(config.dataset_hash)
    bars = symbol_frame(dataset, config.symbol)
    features = load_features(store, config.dataset_hash, config.symbol.upper(), bars)
    mask = pd.Series(True, index=bars.index)
    if config.start_date:
        mask &= bars.index >= pd.Timestamp(config.start_date)
    if config.end_date:
        mask &= bars.index <= pd.Timestamp(config.end_date)
    full = MarketFrame(config.symbol.upper(), bars.loc[mask], features.loc[mask])
    return split_development_test(full, config.split)


def fit_on_development(spec: AgentSpec, dev: DevelopmentData, *, horizon: int,
                       max_seconds: float | None = None) -> BaseTradingAgent:
    """Fit an agent on all usable development rows (labels purged at the dev end)."""
    agent = create_agent(spec.agent_id, spec.genome, spec.seed)
    positions = usable_positions(dev)
    X, y = training_matrix(dev, positions, horizon=horizon)
    agent.fit(TrainingSet(X, y, dev.bars), FitContext(seed=spec.seed, horizon=horizon,
                                                      max_seconds=max_seconds))
    return agent
