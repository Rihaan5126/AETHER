"""Hand-built OHLCV bars for exact simulator tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_bars(opens, closes=None, volume=1_000_000.0, start="2020-01-06") -> pd.DataFrame:
    """Hand-built bars for analytic simulator tests."""
    opens = np.asarray(opens, dtype=float)
    closes = opens.copy() if closes is None else np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(opens))
    high = np.fmax(opens, closes)
    low = np.fmin(opens, closes)
    return pd.DataFrame(
        {"open": opens, "high": high, "low": low, "close": closes, "volume": volume}, index=idx
    )
