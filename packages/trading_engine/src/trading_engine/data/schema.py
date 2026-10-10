"""Canonical OHLCV table layout.

A dataset is a *long* table with one row per (symbol, session date):

==============  =========================================================
column          meaning
==============  =========================================================
timestamp       session date, tz-naive ``datetime64[ns]`` at midnight, in
                the exchange's local calendar (US equities: America/New_York)
symbol          instrument identifier (upper case)
open..close     prices; adjusted for splits and dividends when ``adjustment``
                says so (all four fields use the same adjustment factor)
volume          shares traded (split-adjusted when prices are adjusted)
adjustment      ``"none"`` or ``"split_dividend"``
source          provider name
==============  =========================================================

Engine code that needs a wide per-symbol view uses :func:`symbol_frame`.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

PRICE_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close")
OHLCV_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "symbol",
    *PRICE_COLUMNS,
    "volume",
    "adjustment",
    "source",
)
ADJUSTMENT_VALUES: Final[frozenset[str]] = frozenset({"none", "split_dividend"})


def sort_canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` restricted to canonical columns and sorted by (symbol, timestamp)."""
    out = frame.loc[:, list(OHLCV_COLUMNS)].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.tz_localize(None).dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    for col in (*PRICE_COLUMNS, "volume"):
        out[col] = out[col].astype("float64")
    return out.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)


def symbol_frame(dataset: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return a single symbol's bars indexed by ``timestamp`` (ascending)."""
    sub = dataset.loc[dataset["symbol"] == symbol.upper()]
    if sub.empty:
        raise KeyError(f"symbol {symbol!r} not present in dataset")
    out = sub.set_index("timestamp").sort_index()
    return out.loc[:, [*PRICE_COLUMNS, "volume"]]
