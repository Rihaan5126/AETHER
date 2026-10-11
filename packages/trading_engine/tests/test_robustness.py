"""Validate the robustness tools' assumptions, not just that they run."""

from __future__ import annotations

import math

import numpy as np
import pytest
from bar_factory import make_bars
from trading_engine.research import (
    bootstrap_performance,
    cost_sensitivity,
    null_percentile,
    random_null_sharpes,
    stationary_bootstrap_indices,
)
from trading_engine.simulation import ExecutionConfig


def test_bootstrap_indices_are_valid_and_mostly_contiguous():
    idx = stationary_bootstrap_indices(500, 50, 20.0, np.random.default_rng(0))
    assert idx.shape == (50, 500) and idx.min() >= 0 and idx.max() < 500
    steps = (np.diff(idx, axis=1) % 500) == 1
    # Mean block 20 -> a new block starts with probability 1/20 per step.
    assert 0.93 < steps.mean() < 0.97


def test_sharpe_interval_has_nominal_coverage_on_known_process():
    """90% intervals should contain the true Sharpe about 90% of the time."""
    rng = np.random.default_rng(1)
    mu, sigma, n = 0.0004, 0.01, 750  # true annual Sharpe = 0.635
    true_sharpe = mu / sigma * math.sqrt(252)
    hits = 0
    trials = 150
    for k in range(trials):
        r = rng.normal(mu, sigma, n)
        ci = bootstrap_performance(r, n_boot=400, mean_block=10, level=0.90, seed=k)["sharpe"]
        hits += ci["low"] <= true_sharpe <= ci["high"]
    coverage = hits / trials
    assert 0.80 <= coverage <= 0.97, coverage


def test_blocks_widen_intervals_under_autocorrelation():
    """With positively autocorrelated returns, i.i.d. resampling is overconfident.

    For AR(1) with phi = 0.5 the true standard error of the mean is sqrt(3) = 1.73x
    the i.i.d. one. A single series' bootstrap SE is itself noisy, so average 6.
    """
    rng = np.random.default_rng(2)
    ratios = []
    for k in range(6):
        e = rng.normal(0.0003, 0.01, 1500)
        r = np.empty_like(e)
        r[0] = e[0]
        for t in range(1, len(e)):
            r[t] = 0.5 * r[t - 1] + e[t]
        iid = bootstrap_performance(r, n_boot=600, mean_block=1, seed=k)["annual_return"]
        block = bootstrap_performance(r, n_boot=600, mean_block=40, seed=k)["annual_return"]
        ratios.append((block["high"] - block["low"]) / (iid["high"] - iid["low"]))
    assert 1.3 < float(np.mean(ratios)) < 2.1, ratios


def test_excess_interval_uses_paired_resampling():
    rng = np.random.default_rng(4)
    b = rng.normal(0.0004, 0.01, 1000)
    a = b + 0.0001  # identical risk, tiny constant edge
    out = bootstrap_performance(a, b, n_boot=300, seed=0)
    ex = out["annual_excess_return"]
    # Pairing removes the shared market noise: the edge is estimated almost exactly.
    assert ex["low"] == pytest.approx(0.0252, abs=1e-3)
    assert ex["high"] == pytest.approx(0.0252, abs=1e-3)
    assert ex["prob_positive"] == 1.0


def test_undefined_sharpe_is_none_not_infinite():
    out = bootstrap_performance(np.zeros(300), n_boot=100)
    assert out["sharpe"]["estimate"] is None


def test_random_null_is_deterministic_and_percentile_monotone(dev_test):
    dev, _ = dev_test
    bars, feats = dev.bars.iloc[-400:], dev.features.iloc[-400:]
    a = random_null_sharpes(bars, feats, n=30, seed=7)
    b = random_null_sharpes(bars, feats, n=30, seed=7)
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).sum() >= 25
    assert null_percentile(-10.0, a) == 0.0 and null_percentile(10.0, a) == 1.0
    assert null_percentile(None, a) is None


def test_cost_sensitivity_is_exact_and_monotone():
    bars = make_bars(list(np.linspace(100, 120, 60)))
    weights = np.where(np.arange(60) % 4 < 2, 1.0, 0.0)  # trades often
    rows = cost_sensitivity(bars, weights, ExecutionConfig(fee_bps=5, slippage_bps=5))
    assert [r["cost_multiplier"] for r in rows] == [0.0, 1.0, 2.0, 5.0]
    assert rows[0]["cost_drag"] == 0.0
    returns = [r["total_return"] for r in rows]
    assert returns == sorted(returns, reverse=True) and returns[0] > returns[-1]


def test_shared_indices_match_per_call_resampling():
    """The analytics endpoint reuses one resampling across agents; results must not change."""
    rng = np.random.default_rng(9)
    b = rng.normal(0.0004, 0.01, 700)
    a = b + rng.normal(0.0001, 0.004, 700)
    idx = stationary_bootstrap_indices(700, 300, 20, np.random.default_rng(0))
    shared = bootstrap_performance(a, b, mean_block=20, indices=idx)
    fresh = bootstrap_performance(a, b, n_boot=300, mean_block=20, seed=0)
    assert shared == fresh
    with pytest.raises(ValueError):
        bootstrap_performance(a[:-1], b[:-1], indices=idx)
