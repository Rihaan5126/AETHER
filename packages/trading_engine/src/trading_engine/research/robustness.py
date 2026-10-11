"""Statistical robustness checks for simulated strategies.

Three tools, each with a stated assumption:

* :func:`bootstrap_performance` -- stationary block bootstrap (Politis & Romano,
  1994) confidence intervals for annualised Sharpe, annualised return and the
  annualised excess return over a benchmark. Daily returns are serially
  dependent (volatility clusters), so i.i.d. resampling would understate
  uncertainty; resampling random-length blocks (geometric, mean
  ``mean_block``) keeps short-range dependence. Assumes the return process is
  stationary and weakly dependent within the evaluated window. Agent and
  benchmark returns are resampled with the *same* indices, so the excess-return
  interval respects their correlation. Intervals are percentile intervals.
* :func:`random_null_sharpes` -- the Sharpe distribution of seeded random
  long/flat strategies run through the same simulator, costs and window. It
  answers "how good does pure luck look here?" without a distributional
  assumption.
* :func:`cost_sensitivity` -- re-simulates fixed target weights under scaled
  fee/slippage. Exact, because AETHER agents' decisions do not depend on costs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from trading_engine.agents import Observations, create_agent
from trading_engine.agents.genomes import RandomGenome
from trading_engine.evaluation import compute_metrics
from trading_engine.simulation import ExecutionConfig, simulate
from trading_engine.timing import TRADING_DAYS_PER_YEAR

ANN = TRADING_DAYS_PER_YEAR


def stationary_bootstrap_indices(
    n: int, n_boot: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """``(n_boot, n)`` resampling indices with geometric block lengths (circular)."""
    if n < 2:
        raise ValueError("need at least two observations")
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    restart = rng.random((n_boot, n)) < p
    starts = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def _sharpe(r: np.ndarray) -> np.ndarray:
    """Annualised Sharpe along the last axis; NaN where volatility is ~0."""
    sd = r.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 1e-12, r.mean(axis=-1) / sd * math.sqrt(ANN), np.nan)


@dataclass(frozen=True)
class Interval:
    estimate: float | None
    low: float | None
    high: float | None
    prob_positive: float | None  # share of bootstrap samples > 0

    def to_dict(self) -> dict[str, float | None]:
        return {
            k: (None if v is None or not math.isfinite(v) else round(v, 4))
            for k, v in asdict(self).items()
        }


def _interval(point: float, samples: np.ndarray, level: float) -> Interval:
    s = samples[np.isfinite(samples)]
    if not math.isfinite(point) or len(s) < 0.5 * len(samples):
        return Interval(None, None, None, None)
    a = (1 - level) / 2
    return Interval(
        point, float(np.quantile(s, a)), float(np.quantile(s, 1 - a)), float(np.mean(s > 0))
    )


def bootstrap_performance(
    returns: pd.Series | np.ndarray,
    benchmark: pd.Series | np.ndarray | None = None,
    *,
    n_boot: int = 2000,
    mean_block: float = 20.0,
    level: float = 0.95,
    seed: int = 0,
    indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Bootstrap intervals for Sharpe, annual return and (optionally) excess return.

    ``returns`` are daily simple returns after costs; ``benchmark`` must be aligned.
    ``indices`` (from :func:`stationary_bootstrap_indices`) lets callers evaluating
    many strategies on one timeline reuse a single resampling, which is both cheaper
    and gives common random numbers across strategies.
    """
    r = np.asarray(returns, dtype="float64")
    b = None if benchmark is None else np.asarray(benchmark, dtype="float64")
    if b is not None and len(b) != len(r):
        raise ValueError("benchmark returns must align with strategy returns")
    keep = np.isfinite(r) if b is None else np.isfinite(r) & np.isfinite(b)
    r = r[keep]
    b = None if b is None else b[keep]
    if indices is not None and indices.shape[1] != len(r):
        raise ValueError("indices do not match the number of finite observations")
    idx = (
        indices
        if indices is not None
        else stationary_bootstrap_indices(len(r), n_boot, mean_block, np.random.default_rng(seed))
    )
    rb = r[idx]
    out: dict[str, Any] = {
        "sharpe": _interval(float(_sharpe(r)), _sharpe(rb), level).to_dict(),
        "annual_return": _interval(float(r.mean() * ANN), rb.mean(axis=1) * ANN, level).to_dict(),
    }
    if b is not None:
        ex, exb = r - b, rb - b[idx]
        out["annual_excess_return"] = _interval(
            float(ex.mean() * ANN), exb.mean(axis=1) * ANN, level
        ).to_dict()
        out["sharpe_difference"] = _interval(
            float(_sharpe(r) - _sharpe(b)), _sharpe(rb) - _sharpe(b[idx]), level
        ).to_dict()
    out["_meta"] = {"n": len(r), "n_boot": len(idx), "mean_block": mean_block, "level": level}
    return out


def random_null_sharpes(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    *,
    n: int = 200,
    execution: ExecutionConfig | None = None,
    symbol: str = "X",
    seed: int = 0,
) -> np.ndarray:
    """Validation-style Sharpe ratios of ``n`` seeded random long/flat strategies."""
    out = np.full(n, np.nan)
    obs = Observations(symbol, features, bars)
    for i in range(n):
        agent = create_agent(f"null-{i}", RandomGenome(), seed * 1_000_003 + i)
        res = simulate(bars, agent.predict(obs), execution=execution)
        s = compute_metrics(res).sharpe_ratio
        out[i] = np.nan if s is None else s
    return out


def null_percentile(value: float | None, null: np.ndarray) -> float | None:
    """Share of random strategies with a Sharpe at or below ``value``."""
    s = null[np.isfinite(null)]
    if value is None or not len(s):
        return None
    return float(np.mean(s <= value))


def cost_sensitivity(
    bars: pd.DataFrame,
    weights: np.ndarray,
    base: ExecutionConfig,
    multipliers: Sequence[float] = (0.0, 1.0, 2.0, 5.0),
    benchmark_equity: pd.Series | None = None,
) -> list[dict[str, float | None]]:
    """Metrics for the same decisions with fees and slippage scaled by each multiplier."""
    rows = []
    for m in multipliers:
        cfg = base.model_copy(
            update={
                "fee_bps": base.fee_bps * m,
                "slippage_bps": base.slippage_bps * m,
                "fee_fixed": base.fee_fixed * m,
            }
        )
        met = compute_metrics(simulate(bars, weights, execution=cfg), benchmark_equity)
        rows.append(
            {
                "cost_multiplier": m,
                "fee_bps": cfg.fee_bps,
                "slippage_bps": cfg.slippage_bps,
                "total_return": met.total_return,
                "sharpe_ratio": met.sharpe_ratio,
                "excess_return": met.excess_return,
                "cost_drag": met.cost_drag,
            }
        )
    return rows
