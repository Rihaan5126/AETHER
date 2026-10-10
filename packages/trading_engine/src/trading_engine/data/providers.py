"""Market data providers.

Every provider returns the canonical long OHLCV layout from
:mod:`trading_engine.data.schema`. Providers only *fetch*; validation, cleaning and
versioning are separate steps so that every dataset passes the same checks.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from trading_engine.data.schema import OHLCV_COLUMNS, sort_canonical


class MarketDataProvider(ABC):
    """Source of daily OHLCV bars."""

    name: str

    @abstractmethod
    def fetch_ohlcv(self, symbols: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """Return canonical OHLCV rows for ``symbols`` with ``start <= timestamp <= end``."""


class SyntheticMarketSpec(BaseModel):
    """Parameters of the deterministic synthetic market.

    Log returns follow a two-regime (calm / turbulent) Markov-switching process
    with an optional AR(1) component. ``autocorrelation`` plants a *weak, known*
    predictable structure so the ML pipeline can be validated end to end; set it
    to 0 for a pure random walk where no agent should find an edge. Synthetic
    data is a test fixture, not a model of real markets.
    """

    model_config = ConfigDict(frozen=True)

    seed: int = 7
    initial_price: float = Field(100.0, gt=0)
    annual_drift: float = 0.07
    calm_vol: float = Field(0.14, gt=0)
    turbulent_vol: float = Field(0.38, gt=0)
    p_calm_to_turbulent: float = Field(0.01, ge=0, le=1)
    p_turbulent_to_calm: float = Field(0.05, ge=0, le=1)
    autocorrelation: float = Field(0.08, ge=-0.5, le=0.5)
    overnight_share: float = Field(0.3, ge=0, le=1)
    base_volume: float = Field(5_000_000, gt=0)
    correlation_to_first: float = Field(0.6, ge=0, le=1)


def _symbol_seed(seed: int, symbol: str) -> int:
    # Stable across processes (unlike hash()).
    return seed * 1_000_003 + sum((i + 1) * ord(c) for i, c in enumerate(symbol.upper()))


class SyntheticDataProvider(MarketDataProvider):
    """Deterministic synthetic OHLCV generator for offline development and tests.

    Calendar: Monday-Friday business days (no holidays). Identical inputs always
    produce byte-identical output, which the dataset hash relies on.
    """

    name = "synthetic"

    def __init__(self, spec: SyntheticMarketSpec | None = None) -> None:
        self.spec = spec or SyntheticMarketSpec()

    def fetch_ohlcv(self, symbols: Sequence[str], start: date, end: date) -> pd.DataFrame:
        if not symbols:
            raise ValueError("at least one symbol is required")
        dates = pd.bdate_range(start=start, end=end)
        if len(dates) < 2:
            raise ValueError("date range must contain at least two business days")
        spec = self.spec
        n = len(dates)
        dt = 1.0 / 252.0

        # Shared market factor so multiple symbols are correlated.
        market_rng = np.random.default_rng(_symbol_seed(spec.seed, "__MARKET__"))
        regimes = self._simulate_regimes(market_rng, n)
        market_shocks = market_rng.standard_normal(n)

        frames = []
        for symbol in symbols:
            rng = np.random.default_rng(_symbol_seed(spec.seed, symbol))
            idio = rng.standard_normal(n)
            rho = spec.correlation_to_first
            shocks = rho * market_shocks + math.sqrt(1 - rho**2) * idio
            vol = np.where(regimes == 1, spec.turbulent_vol, spec.calm_vol) * math.sqrt(dt)
            drift = (spec.annual_drift - 0.5 * (vol / math.sqrt(dt)) ** 2) * dt
            log_ret = np.empty(n)
            prev = 0.0
            for i in range(n):
                prev = drift[i] + spec.autocorrelation * (prev - drift[i]) + vol[i] * shocks[i]
                log_ret[i] = prev
            log_ret[0] = 0.0
            frames.append(self._bars_from_returns(rng, symbol, dates, log_ret, vol, regimes))
        return sort_canonical(pd.concat(frames, ignore_index=True))

    def _simulate_regimes(self, rng: np.random.Generator, n: int) -> np.ndarray:
        spec = self.spec
        regimes = np.zeros(n, dtype=np.int8)
        u = rng.random(n)
        for i in range(1, n):
            if regimes[i - 1] == 0:
                regimes[i] = 1 if u[i] < spec.p_calm_to_turbulent else 0
            else:
                regimes[i] = 0 if u[i] < spec.p_turbulent_to_calm else 1
        return regimes

    def _bars_from_returns(
        self,
        rng: np.random.Generator,
        symbol: str,
        dates: pd.DatetimeIndex,
        log_ret: np.ndarray,
        vol: np.ndarray,
        regimes: np.ndarray,
    ) -> pd.DataFrame:
        spec = self.spec
        close = spec.initial_price * np.exp(np.cumsum(log_ret))
        prev_close = np.concatenate([[spec.initial_price], close[:-1]])
        # Split each day's move into an overnight gap and an intraday move.
        overnight = spec.overnight_share * log_ret
        open_ = prev_close * np.exp(overnight)
        open_[0] = spec.initial_price
        wick_up = np.abs(rng.standard_normal(len(dates))) * vol * 0.5
        wick_dn = np.abs(rng.standard_normal(len(dates))) * vol * 0.5
        high = np.maximum(open_, close) * np.exp(wick_up)
        low = np.minimum(open_, close) * np.exp(-wick_dn)
        volume_noise = rng.lognormal(mean=0.0, sigma=0.25, size=len(dates))
        volume = spec.base_volume * volume_noise * (1 + 25 * np.abs(log_ret)) * (1 + 0.5 * regimes)
        return pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": symbol.upper(),
                "open": np.round(open_, 6),
                "high": np.round(high, 6),
                "low": np.round(low, 6),
                "close": np.round(close, 6),
                "volume": np.round(volume),
                "adjustment": "none",
                "source": self.name,
            }
        )


class FileDataProvider(MarketDataProvider):
    """Load OHLCV bars from a local CSV or Parquet file.

    Accepted layouts: the canonical long layout, or a single-symbol file without a
    ``symbol`` column (pass ``default_symbol``). Column names are matched
    case-insensitively; ``date`` is accepted as an alias of ``timestamp`` and
    ``adj close`` is ignored unless ``use_adjusted_close`` is set, in which case all
    OHLC fields are scaled by ``adj_close / close``.
    """

    name = "file"
    MAX_BYTES = 200 * 1024 * 1024

    def __init__(
        self,
        path: str | Path,
        *,
        default_symbol: str | None = None,
        use_adjusted_close: bool = False,
    ) -> None:
        self.path = Path(path)
        self.default_symbol = default_symbol
        self.use_adjusted_close = use_adjusted_close

    def read_all(self) -> pd.DataFrame:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if self.path.stat().st_size > self.MAX_BYTES:
            raise ValueError(f"file exceeds {self.MAX_BYTES} bytes")
        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            raw = pd.read_csv(self.path)
        elif suffix in {".parquet", ".pq"}:
            raw = pd.read_parquet(self.path)
        else:
            raise ValueError(f"unsupported file type {suffix!r}; use .csv or .parquet")
        return self._normalise(raw)

    def _normalise(self, raw: pd.DataFrame) -> pd.DataFrame:
        cols = {c: c.strip().lower().replace("_", " ") for c in raw.columns}
        frame = raw.rename(columns=cols)
        frame = frame.rename(columns={"date": "timestamp", "adj close": "adj_close"})
        if "symbol" not in frame.columns:
            if not self.default_symbol:
                raise ValueError("file has no 'symbol' column and no default_symbol was given")
            frame["symbol"] = self.default_symbol
        missing = {"timestamp", "open", "high", "low", "close", "volume"} - set(frame.columns)
        if missing:
            raise ValueError(f"file is missing required columns: {sorted(missing)}")
        adjustment = "none"
        if self.use_adjusted_close:
            if "adj_close" not in frame.columns:
                raise ValueError("use_adjusted_close requires an 'Adj Close' column")
            factor = frame["adj_close"].astype(float) / frame["close"].astype(float)
            for col in ("open", "high", "low", "close"):
                frame[col] = frame[col].astype(float) * factor
            frame["volume"] = frame["volume"].astype(float) / factor
            adjustment = "split_dividend"
        if "adjustment" not in frame.columns:
            frame["adjustment"] = adjustment
        frame["source"] = frame["source"] if "source" in frame.columns else f"file:{self.path.name}"
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=False)
        if getattr(frame["timestamp"].dt, "tz", None) is not None:
            frame["timestamp"] = frame["timestamp"].dt.tz_convert("America/New_York")
        return sort_canonical(frame.loc[:, list(OHLCV_COLUMNS)])

    def fetch_ohlcv(self, symbols: Sequence[str], start: date, end: date) -> pd.DataFrame:
        frame = self.read_all()
        wanted = {s.upper() for s in symbols}
        mask = (
            frame["symbol"].isin(wanted)
            & (frame["timestamp"] >= pd.Timestamp(start))
            & (frame["timestamp"] <= pd.Timestamp(end))
        )
        return frame.loc[mask].reset_index(drop=True)


class YahooFinanceProvider(MarketDataProvider):
    """Optional public provider using the ``yfinance`` package (prototyping only).

    Yahoo data is unofficial, may change or disappear, has no point-in-time
    constituents and is subject to Yahoo's terms of use. AETHER only calls the
    documented ``yfinance.download`` function at a modest rate and never runs it in
    tests. Install with ``uv sync --extra yahoo``.
    """

    name = "yahoo"

    def fetch_ohlcv(self, symbols: Sequence[str], start: date, end: date) -> pd.DataFrame:
        try:
            import yfinance as yf  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("yfinance is not installed; run `uv sync --extra yahoo`") from exc
        frames = []
        for symbol in symbols:
            raw = yf.download(
                symbol,
                start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                actions=False,
            )
            if raw is None or raw.empty:
                raise RuntimeError(f"no data returned for {symbol}")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.reset_index().rename(columns=str.lower).rename(columns={"date": "timestamp"})
            raw["symbol"] = symbol.upper()
            raw["adjustment"] = "split_dividend"
            raw["source"] = self.name
            frames.append(raw)
        return sort_canonical(pd.concat(frames, ignore_index=True))
