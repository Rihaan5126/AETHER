"""Market data: providers, validation and immutable dataset versions."""

from trading_engine.data.providers import (
    FileDataProvider,
    MarketDataProvider,
    SyntheticDataProvider,
    SyntheticMarketSpec,
    YahooFinanceProvider,
)
from trading_engine.data.schema import OHLCV_COLUMNS, PRICE_COLUMNS
from trading_engine.data.validation import (
    DataQualityError,
    ValidationIssue,
    ValidationReport,
    clean_ohlcv,
    validate_ohlcv,
)
from trading_engine.data.versioning import DatasetManifest, DatasetStore, compute_dataset_hash

__all__ = [
    "OHLCV_COLUMNS",
    "PRICE_COLUMNS",
    "DataQualityError",
    "DatasetManifest",
    "DatasetStore",
    "FileDataProvider",
    "MarketDataProvider",
    "SyntheticDataProvider",
    "SyntheticMarketSpec",
    "ValidationIssue",
    "ValidationReport",
    "YahooFinanceProvider",
    "clean_ohlcv",
    "compute_dataset_hash",
    "validate_ohlcv",
]
