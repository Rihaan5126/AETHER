from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trading_engine.data import SyntheticDataProvider
from trading_engine.data.schema import symbol_frame
from trading_engine.features import compute_features
from trading_engine.training import MarketFrame, SplitConfig, split_development_test


@pytest.fixture(scope="session")
def synthetic_dataset() -> pd.DataFrame:
    return SyntheticDataProvider().fetch_ohlcv(["SYN"], date(2014, 1, 1), date(2021, 12, 31))


@pytest.fixture(scope="session")
def bars(synthetic_dataset: pd.DataFrame) -> pd.DataFrame:
    return symbol_frame(synthetic_dataset, "SYN")


@pytest.fixture(scope="session")
def market(bars: pd.DataFrame) -> MarketFrame:
    return MarketFrame("SYN", bars, compute_features(bars))


@pytest.fixture(scope="session")
def dev_test(market: MarketFrame):
    return split_development_test(market, SplitConfig(test_fraction=0.2))


def make_bars(opens, closes=None, volume=1_000_000.0, start="2020-01-06") -> pd.DataFrame:
    """Hand-built bars for analytic simulator tests."""
    opens = np.asarray(opens, dtype=float)
    closes = opens.copy() if closes is None else np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(opens))
    high = np.fmax(opens, closes)
    low = np.fmin(opens, closes)
    return pd.DataFrame({"open": opens, "high": high, "low": low, "close": closes,
                         "volume": volume}, index=idx)
