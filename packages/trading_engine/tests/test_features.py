from __future__ import annotations

import numpy as np
import pandas as pd

from trading_engine.features import FEATURE_SPECS, compute_features, feature_names, max_lookback


def test_features_never_use_future_bars(bars):
    full = compute_features(bars)
    for cut in (120, 400, 900, len(bars) - 1):
        partial = compute_features(bars.iloc[: cut + 1])
        pd.testing.assert_frame_equal(full.iloc[: cut + 1], partial, check_exact=False,
                                      rtol=1e-9, atol=1e-12)


def test_poisoned_future_does_not_change_past_features(bars):
    poisoned = bars.copy()
    poisoned.iloc[600:, :] = poisoned.iloc[600:, :] * 3.0
    a = compute_features(bars).iloc[:600]
    b = compute_features(poisoned).iloc[:600]
    pd.testing.assert_frame_equal(a, b)


def test_rolling_calculations_match_manual(bars):
    f = compute_features(bars)
    t = 300
    close = bars["close"]
    assert np.isclose(f["sma_20_ratio"].iloc[t], close.iloc[t] / close.iloc[t - 19: t + 1].mean() - 1)
    assert np.isclose(f["ret_5"].iloc[t], close.iloc[t] / close.iloc[t - 5] - 1)
    rets = close.pct_change().iloc[t - 19: t + 1]
    assert np.isclose(f["vol_20"].iloc[t], rets.std() * np.sqrt(252))
    assert np.isclose(f["drawdown_60"].iloc[t], close.iloc[t] / close.iloc[t - 59: t + 1].max() - 1)


def test_nan_handling_and_bounds(bars):
    f = compute_features(bars)
    assert list(f.columns) == feature_names()
    assert f.iloc[max_lookback():].notna().all().all()
    assert f["sma_50_ratio"].iloc[:49].isna().all()
    rsi = f["rsi_14"].dropna()
    assert ((rsi >= 0) & (rsi <= 1)).all()
    assert not np.isinf(f.to_numpy(dtype=float)).any()


def test_features_are_deterministic(bars):
    pd.testing.assert_frame_equal(compute_features(bars), compute_features(bars))


def test_every_feature_documents_availability():
    assert all(s.available_at == "close" and s.lookback >= 1 for s in FEATURE_SPECS)
