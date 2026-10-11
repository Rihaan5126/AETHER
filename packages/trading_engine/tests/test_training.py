from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from trading_engine.training import (
    DevelopmentData,
    MarketFrame,
    SplitConfig,
    WalkForwardConfig,
    forward_open_return,
    make_direction_labels,
    split_development_test,
    walk_forward_folds,
)
from trading_engine.training.splits import training_matrix


def test_label_matches_executable_holding_period(bars):
    fwd = forward_open_return(bars, horizon=1)
    t = 100
    expected = bars["open"].iloc[t + 2] / bars["open"].iloc[t + 1] - 1
    assert np.isclose(fwd.iloc[t], expected)
    # Not the close-to-close return that would require trading at close[t].
    assert not np.isclose(fwd.iloc[t], bars["close"].iloc[t + 1] / bars["close"].iloc[t] - 1)
    labels = make_direction_labels(bars, horizon=3)
    assert labels.iloc[-4:].isna().all() and labels.iloc[:-4].notna().all()


def test_development_data_excludes_test_period(dev_test):
    dev, test = dev_test
    assert dev.end < test.start == dev.test_start
    assert len(dev.index.intersection(test.index)) == 0
    with pytest.raises(ValueError, match="overlaps"):
        DevelopmentData(dev.symbol, test.bars, test.features, test_start=test.start)


def test_explicit_test_start(market):
    dev, test = split_development_test(market, SplitConfig(test_start=pd.Timestamp("2020-06-01")))
    assert test.start >= pd.Timestamp("2020-06-01") and dev.end < pd.Timestamp("2020-06-01")


def test_walk_forward_folds_are_chronological_and_purged(dev_test):
    dev, _ = dev_test
    cfg = WalkForwardConfig(n_folds=3, embargo_bars=5, horizon=1)
    folds = walk_forward_folds(dev, cfg)
    assert len(folds) == 3
    for prev, nxt in itertools.pairwise(folds):
        assert prev.val_stop <= nxt.val_start
    for fold in folds:
        # last training label window (t+1+h) ends strictly before validation starts
        assert fold.train_positions[-1] + 1 + cfg.horizon < fold.val_start
        assert fold.val_start - fold.train_positions[-1] - 1 >= cfg.horizon + 1 + cfg.embargo_bars


def test_rolling_window_mode(dev_test):
    dev, _ = dev_test
    folds = walk_forward_folds(dev, WalkForwardConfig(mode="rolling", rolling_window=300))
    assert all(len(f.train_positions) <= 300 for f in folds)


def test_training_labels_do_not_see_validation_prices(dev_test):
    dev, _ = dev_test
    fold = walk_forward_folds(dev, WalkForwardConfig())[0]
    X, y = training_matrix(dev, fold.train_positions, horizon=1, label_cutoff=fold.val_start)
    poisoned_bars = dev.bars.copy()
    poisoned_bars.iloc[fold.val_start :] *= 10
    poisoned = MarketFrame(dev.symbol, poisoned_bars, dev.features)
    X2, y2 = training_matrix(poisoned, fold.train_positions, horizon=1, label_cutoff=fold.val_start)
    pd.testing.assert_series_equal(y, y2)
    pd.testing.assert_frame_equal(X, X2)
    assert X.index.max() < dev.index[fold.val_start]
