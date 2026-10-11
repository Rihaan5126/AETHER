"""End-to-end look-ahead tests at the agent and portfolio level.

The feature tests prove features at day t ignore later bars. These go further:
an agent's *decision* for day t, and the simulated portfolio up to t, must be
identical whether the bars after t are real or deliberately corrupted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from bar_factory import make_bars
from trading_engine.agents import FitContext, Observations, TrainingSet, create_agent, parse_genome
from trading_engine.features import compute_features
from trading_engine.simulation import ExecutionConfig, simulate
from trading_engine.tournaments.engine import stitch
from trading_engine.training import WalkForwardConfig, walk_forward_folds
from trading_engine.training.splits import training_matrix

FAMILIES = ["random", "buy_and_hold", "momentum", "random_forest", "xgboost", "mlp"]
FAST = {
    "random_forest": {"n_estimators": 60},
    "xgboost": {"n_estimators": 60},
    "mlp": {"max_epochs": 8},
}


def _poison_after(bars: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    """Replace every bar after ``cut`` with a wildly different (but valid) path."""
    out = bars.copy()
    after = out.index > cut
    rng = np.random.default_rng(99)
    shock = np.exp(np.cumsum(rng.normal(0, 0.05, after.sum())))[:, None]
    out.loc[after, ["open", "high", "low", "close"]] = (
        out.loc[after, ["open", "high", "low", "close"]].to_numpy() * shock * 1.7
    )
    out.loc[after, "volume"] = out.loc[after, "volume"] * 5
    return out


@pytest.fixture(scope="module")
def setup(dev_test):
    dev, _ = dev_test
    fold = walk_forward_folds(dev, WalkForwardConfig(n_folds=2))[1]
    X, y = training_matrix(dev, fold.train_positions, horizon=1, label_cutoff=fold.val_start)
    training = TrainingSet(X, y, dev.bars.iloc[: fold.val_start])
    val_index = dev.index[fold.val_start : fold.val_stop]
    cut = val_index[len(val_index) // 2]
    return dev, fold, training, val_index, cut


@pytest.mark.parametrize("family", FAMILIES)
def test_decisions_up_to_t_ignore_bars_after_t(family, setup):
    dev, fold, training, val_index, cut = setup
    agent = create_agent(
        f"la-{family}", parse_genome({"family": family, **FAST.get(family, {})}), 5
    )
    agent.fit(training, FitContext(seed=5))

    real_bars = dev.bars.iloc[: fold.val_stop]
    real = Observations(dev.symbol, dev.features.loc[val_index], real_bars)

    poisoned_bars = _poison_after(real_bars, cut)
    poisoned_feats = compute_features(poisoned_bars).loc[val_index]
    poisoned = Observations(dev.symbol, poisoned_feats, poisoned_bars)

    w_real = agent.predict(real)
    w_poisoned = agent.predict(poisoned)
    upto = val_index <= cut
    np.testing.assert_allclose(w_poisoned[upto], w_real[upto], rtol=0, atol=1e-12)


def test_portfolio_up_to_t_ignores_prices_after_t(setup):
    """Same weights, corrupted future: equity and trades through day t are unchanged."""
    dev, _, _, val_index, cut = setup
    bars = dev.bars.loc[val_index]
    weights = np.where(np.arange(len(bars)) % 7 < 4, 1.0, 0.3)
    a = simulate(bars, weights)
    b = simulate(_poison_after(bars, cut), weights)
    upto = a.equity.index <= cut
    pd.testing.assert_frame_equal(a.equity.loc[upto], b.equity.loc[upto])
    executed_by_cut = [t for t in a.trades if pd.Timestamp(t.executed_time) <= cut]
    assert executed_by_cut == b.trades[: len(executed_by_cut)]


def test_stitched_validation_curve_compounds_fold_returns():
    """Folds start from cash; the stitched curve must equal the product of fold returns."""
    cfg = ExecutionConfig(fee_bps=10, slippage_bps=5)
    r1 = simulate(make_bars([100, 102, 104, 101, 106]), np.ones(5), execution=cfg)
    r2 = simulate(make_bars([50, 49, 47, 48, 52], start="2020-03-02"), np.ones(5), execution=cfg)
    fold_returns = [res.final_equity / cfg.initial_capital for res in (r1, r2)]
    stitched = stitch([r1, r2], cfg.initial_capital)
    expected = cfg.initial_capital * np.prod(fold_returns)
    assert stitched.final_equity == pytest.approx(expected, rel=1e-12)
    # Trade ledger is rescaled consistently: fees stay the same fraction of notional.
    for t in stitched.trades:
        assert t.fee == pytest.approx(t.notional * 10 / 1e4, rel=1e-9)
    assert (stitched.equity["drawdown"] <= 1e-12).all()
