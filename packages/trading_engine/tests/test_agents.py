from __future__ import annotations

import numpy as np
import pydantic
import pytest
from trading_engine.agents import (
    FitContext,
    Observations,
    TrainingSet,
    create_agent,
    default_genome,
    load_agent,
    parse_genome,
    save_agent,
)
from trading_engine.agents.genomes import MLPGenome, MomentumGenome, XGBoostGenome
from trading_engine.storage import ArtifactIntegrityError, LocalArtifactStore
from trading_engine.training import WalkForwardConfig, walk_forward_folds
from trading_engine.training.splits import training_matrix

ALL = ["random", "buy_and_hold", "momentum", "random_forest", "xgboost", "mlp"]
LEARNED = ["random_forest", "xgboost", "mlp"]
FAST = {
    "random_forest": {"n_estimators": 60},
    "xgboost": {"n_estimators": 60},
    "mlp": {"max_epochs": 15},
}


@pytest.fixture(scope="module")
def fold_data(dev_test):
    dev, _ = dev_test
    fold = walk_forward_folds(dev, WalkForwardConfig(n_folds=2))[0]
    X, y = training_matrix(dev, fold.train_positions, horizon=1, label_cutoff=fold.val_start)
    val = dev.slice_positions(fold.val_start, fold.val_stop)
    obs = Observations(dev.symbol, val.features, dev.bars.iloc[: fold.val_stop])
    return TrainingSet(X, y, dev.bars.iloc[: fold.val_start]), obs


def make(family: str, seed: int = 3):
    genome = parse_genome({"family": family, **FAST.get(family, {})})
    return create_agent(f"t-{family}", genome, seed)


@pytest.mark.parametrize("family", ALL)
def test_agents_produce_valid_weights(family, fold_data):
    training, obs = fold_data
    agent = make(family)
    agent.fit(training, FitContext(seed=3))
    w = agent.predict(obs)
    assert w.shape == (len(obs.features),)
    assert not np.isnan(w).any()
    assert ((w >= 0) & (w <= 1)).all()
    decisions = agent.decisions(obs)
    assert len(decisions) == len(w) and decisions[0].symbol == "SYN"


@pytest.mark.parametrize("family", LEARNED)
def test_learned_agents_really_fit(family, fold_data):
    training, obs = fold_data
    agent = make(family)
    with pytest.raises(RuntimeError, match="fitted"):
        agent.predict(obs)
    metrics = agent.fit(training, FitContext(seed=3))
    assert metrics["fitted"] and metrics["n_samples"] == len(training.X)
    assert metrics["train_end"] < str(obs.features.index[0].date())
    scores = agent.forecast(obs)
    assert scores.std() > 0  # not a constant output


@pytest.mark.parametrize("family", LEARNED)
def test_training_is_deterministic(family, fold_data):
    training, obs = fold_data
    a, b = make(family), make(family)
    a.fit(training, FitContext(seed=3))
    b.fit(training, FitContext(seed=3))
    np.testing.assert_allclose(a.forecast(obs), b.forecast(obs), rtol=1e-6)


@pytest.mark.parametrize("family", ALL)
def test_save_load_roundtrip(family, fold_data, tmp_path):
    training, obs = fold_data
    agent = make(family)
    agent.fit(training, FitContext(seed=3))
    store = LocalArtifactStore(tmp_path)
    ref = save_agent(agent, store, f"agents/{family}")
    restored = load_agent(store, ref)
    np.testing.assert_allclose(agent.predict(obs), restored.predict(obs), rtol=1e-6)


def test_tampered_artifact_is_never_loaded(fold_data, tmp_path):
    training, _ = fold_data
    agent = make("random_forest")
    agent.fit(training, FitContext(seed=3))
    store = LocalArtifactStore(tmp_path)
    ref = save_agent(agent, store, "agents/rf")
    (tmp_path / "agents/rf/model.joblib").write_bytes(b"not a model")
    with pytest.raises(ArtifactIntegrityError):
        load_agent(store, ref)


def test_artifact_keys_cannot_escape_store(tmp_path):
    store = LocalArtifactStore(tmp_path / "store")
    with pytest.raises(ValueError):
        store.workdir("../outside")


def test_random_agent_is_reproducible(fold_data):
    _, obs = fold_data
    a, b = make("random", seed=11), make("random", seed=11)
    np.testing.assert_array_equal(a.predict(obs), b.predict(obs))
    assert not np.array_equal(a.predict(obs), make("random", seed=12).predict(obs))


def test_genome_validation():
    with pytest.raises(pydantic.ValidationError):
        MomentumGenome(fast_window=50, slow_window=40)
    with pytest.raises(pydantic.ValidationError):
        XGBoostGenome(max_depth=50)
    with pytest.raises(pydantic.ValidationError):
        XGBoostGenome(features=("ret_1", "not_a_feature", "ret_5"))
    with pytest.raises(pydantic.ValidationError):
        MLPGenome(hidden_layers=(8, 8, 8, 8))
    assert parse_genome(default_genome("mlp").model_dump(mode="json")) == default_genome("mlp")


def test_agent_rejects_wrong_genome():
    from trading_engine.agents import RandomForestAgent

    with pytest.raises(TypeError):
        RandomForestAgent("x", default_genome("xgboost"), 1)  # type: ignore[arg-type]
