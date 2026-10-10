from __future__ import annotations

from trading_engine.optimization import OptimizationConfig, optimize
from trading_engine.tournaments import EvaluationConfig
from trading_engine.training import WalkForwardConfig

EVAL = EvaluationConfig(walk_forward=WalkForwardConfig(n_folds=3))


def test_optuna_study_is_bounded_and_reproducible(dev_test):
    dev, _ = dev_test
    cfg = OptimizationConfig(family="momentum", n_trials=6, seed=5)
    a = optimize(dev, EVAL, cfg)
    b = optimize(dev, EVAL, cfg)
    assert len(a.trials) == 6
    assert [t.params for t in a.trials] == [t.params for t in b.trials]
    assert a.best_value == b.best_value
    assert {t.state for t in a.trials} <= {"complete", "pruned", "failed"}
    assert a.best_genome is not None and a.best_genome["family"] == "momentum"
