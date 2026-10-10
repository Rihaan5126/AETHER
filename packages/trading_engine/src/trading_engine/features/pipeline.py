"""Feature pipeline.

Every feature for session ``t`` is computed from bars ``<= t`` only and becomes
available after the close of ``t`` (see :mod:`trading_engine.timing`). All
operations are trailing (``rolling``, ``shift(k>=0)``, recursive ``ewm`` with
``adjust=False``). Features are expressed as scale-free quantities (returns,
ratios, z-scores) so that models trained on one price level transfer to another.

No normalisation happens here: scalers are learned inside each agent's model
pipeline on the training window only. Leading rows without enough history are
``NaN``; consumers drop them (``max_lookback()`` rows).

The test-suite verifies the point-in-time property by recomputing features on
truncated histories and checking that past values never change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from trading_engine.timing import FEATURE_AVAILABILITY, TRADING_DAYS_PER_YEAR

FeatureGroup = Literal["returns", "trend", "volatility", "momentum", "volume", "regime"]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: FeatureGroup
    lookback: int
    description: str
    available_at: str = FEATURE_AVAILABILITY


FEATURE_SPECS: Final[tuple[FeatureSpec, ...]] = (
    FeatureSpec("ret_1", "returns", 1, "1-session close-to-close return"),
    FeatureSpec("ret_5", "returns", 5, "5-session return"),
    FeatureSpec("ret_10", "returns", 10, "10-session return"),
    FeatureSpec("ret_20", "returns", 20, "20-session return"),
    FeatureSpec("sma_10_ratio", "trend", 10, "close / SMA(10) - 1"),
    FeatureSpec("sma_20_ratio", "trend", 20, "close / SMA(20) - 1"),
    FeatureSpec("sma_50_ratio", "trend", 50, "close / SMA(50) - 1"),
    FeatureSpec("ema_12_ratio", "trend", 12, "close / EMA(12) - 1"),
    FeatureSpec("ema_26_ratio", "trend", 26, "close / EMA(26) - 1"),
    FeatureSpec("sma_10_50_spread", "trend", 50, "SMA(10) / SMA(50) - 1"),
    FeatureSpec("vol_20", "volatility", 21, "annualised std of daily returns, 20 sessions"),
    FeatureSpec("vol_60", "volatility", 61, "annualised realised volatility, 60 sessions"),
    FeatureSpec("atr_14_pct", "volatility", 15, "Wilder ATR(14) / close"),
    FeatureSpec("range_20", "volatility", 20, "(20-session high - low) / close"),
    FeatureSpec("rsi_14", "momentum", 15, "Wilder RSI(14) scaled to [0, 1]"),
    FeatureSpec("macd", "momentum", 26, "(EMA12 - EMA26) / close"),
    FeatureSpec("macd_hist", "momentum", 35, "MACD minus its 9-session EMA, / close"),
    FeatureSpec("mom_60", "momentum", 60, "60-session return"),
    FeatureSpec("drawdown_60", "momentum", 60, "close / 60-session max close - 1"),
    FeatureSpec("rel_volume_20", "volume", 20, "volume / 20-session mean volume"),
    FeatureSpec("volume_change_5", "volume", 20, "log(mean vol 5 / mean vol 20)"),
    FeatureSpec("bench_vol_60", "regime", 61, "benchmark 60-session realised volatility"),
    FeatureSpec("bench_mom_60", "regime", 60, "benchmark 60-session return"),
    FeatureSpec("trend_strength", "regime", 21, "|20-session return| / (vol_20 * sqrt(20/252))"),
    FeatureSpec("rel_ret_20", "regime", 20, "asset ret_20 minus benchmark ret_20"),
)

FEATURE_GROUPS: Final[dict[str, list[str]]] = {}
for _spec in FEATURE_SPECS:
    FEATURE_GROUPS.setdefault(_spec.group, []).append(_spec.name)


def feature_names() -> list[str]:
    return [s.name for s in FEATURE_SPECS]


def max_lookback() -> int:
    """Warm-up rows (with margin for recursive EMA/RSI initialisation)."""
    return max(s.lookback for s in FEATURE_SPECS) + 5


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 1 - 1 / (1 + rs)
    # All gains and no losses -> RSI = 1; flat -> 0.5.
    rsi = rsi.where(loss != 0, np.where(gain > 0, 1.0, 0.5))
    return rsi.where(gain.notna())


def _atr(bars: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = bars["close"].shift(1)
    tr = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def compute_features(bars: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute all features for one symbol.

    ``bars`` is indexed by session date with open/high/low/close/volume columns.
    ``benchmark`` (same layout) supplies regime features; when ``None`` the asset is
    its own benchmark (single-asset MVP).
    """
    if not bars.index.is_monotonic_increasing:
        raise ValueError("bars must be sorted by timestamp")
    close = bars["close"]
    ret = close.pct_change(fill_method=None)
    ann = np.sqrt(TRADING_DAYS_PER_YEAR)
    f = pd.DataFrame(index=bars.index)

    for k in (1, 5, 10, 20):
        f[f"ret_{k}"] = close.pct_change(k, fill_method=None)
    sma10, sma20, sma50 = (close.rolling(w).mean() for w in (10, 20, 50))
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    f["sma_10_ratio"] = close / sma10 - 1
    f["sma_20_ratio"] = close / sma20 - 1
    f["sma_50_ratio"] = close / sma50 - 1
    f["ema_12_ratio"] = close / ema12 - 1
    f["ema_26_ratio"] = close / ema26 - 1
    f["sma_10_50_spread"] = sma10 / sma50 - 1

    f["vol_20"] = ret.rolling(20).std() * ann
    f["vol_60"] = ret.rolling(60).std() * ann
    f["atr_14_pct"] = _atr(bars) / close
    f["range_20"] = (bars["high"].rolling(20).max() - bars["low"].rolling(20).min()) / close

    f["rsi_14"] = _rsi(close)
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    f["macd"] = macd_line / close
    f["macd_hist"] = (macd_line - signal) / close
    f["mom_60"] = close.pct_change(60, fill_method=None)
    f["drawdown_60"] = close / close.rolling(60).max() - 1

    volume = bars["volume"].replace(0, np.nan)
    vol_mean20 = volume.rolling(20).mean()
    f["rel_volume_20"] = volume / vol_mean20
    f["volume_change_5"] = np.log(volume.rolling(5).mean() / vol_mean20)

    bench = benchmark if benchmark is not None else bars
    bench_close = bench["close"].reindex(bars.index)
    bench_ret = bench_close.pct_change(fill_method=None)
    f["bench_vol_60"] = bench_ret.rolling(60).std() * ann
    f["bench_mom_60"] = bench_close.pct_change(60, fill_method=None)
    f["trend_strength"] = f["ret_20"].abs() / (f["vol_20"] * np.sqrt(20 / TRADING_DAYS_PER_YEAR))
    f["rel_ret_20"] = f["ret_20"] - bench_close.pct_change(20, fill_method=None)

    f = f.replace([np.inf, -np.inf], np.nan)
    return f.loc[:, feature_names()]
