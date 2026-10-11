from __future__ import annotations

from datetime import date

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
