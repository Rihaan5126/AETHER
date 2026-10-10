"""Chronological development/test split and walk-forward folds.

Timeline::

    |<---------------- development ---------------->|<-- final test -->|
    | warm-up | fold-1 train | gap | fold-1 val      |                  |
    | warm-up | fold-2 train (expanding)  | gap | val |                  |

* The final test period is cut off first. :class:`DevelopmentData` physically
  contains no bars on or after ``test_start``; everything that selects, tunes or
  evolves agents receives only this object.
* Inside development, folds are walk-forward: train on the past, validate on the
  next unseen window. Training labels are computed on bars truncated at the
  validation start, so any label whose holding window would reach validation
  prices is ``NaN`` and dropped (purging); ``embargo_bars`` adds an extra gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading_engine.training.labels import make_direction_labels


@dataclass(frozen=True)
class MarketFrame:
    """Bars and point-in-time features for one symbol over one period."""

    symbol: str
    bars: pd.DataFrame
    features: pd.DataFrame
    period: Literal["development", "test", "full"] = "full"

    def __post_init__(self) -> None:
        if not self.bars.index.equals(self.features.index):
            raise ValueError("bars and features must share the same index")

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.bars.index)

    @property
    def start(self) -> pd.Timestamp:
        return self.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.index[-1]

    def slice_positions(self, start: int, stop: int) -> MarketFrame:
        return MarketFrame(
            self.symbol, self.bars.iloc[start:stop], self.features.iloc[start:stop], self.period
        )


@dataclass(frozen=True)
class DevelopmentData(MarketFrame):
    """Data allowed for training, validation, selection and evolution."""

    test_start: pd.Timestamp | None = None
    period: Literal["development", "test", "full"] = "development"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.test_start is not None and len(self.index) and self.end >= self.test_start:
            raise ValueError("development data overlaps the final test period")


@dataclass(frozen=True)
class TestData(MarketFrame):
    """Locked final-test period. Only the final evaluation procedure may use it."""

    period: Literal["development", "test", "full"] = "test"


class SplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    test_start: date | None = None
    test_fraction: float = Field(0.2, gt=0, lt=0.9)

    def resolve_test_start(self, index: pd.DatetimeIndex) -> pd.Timestamp:
        if self.test_start is not None:
            ts = pd.Timestamp(self.test_start)
            if ts <= index[0] or ts > index[-1]:
                raise ValueError("test_start must fall inside the dataset")
            return index[index >= ts][0]
        pos = round(len(index) * (1 - self.test_fraction))
        return index[min(max(pos, 1), len(index) - 1)]


def split_development_test(
    full: MarketFrame, config: SplitConfig
) -> tuple[DevelopmentData, TestData]:
    test_start = config.resolve_test_start(full.index)
    dev_mask = full.index < test_start
    dev = DevelopmentData(
        full.symbol, full.bars.loc[dev_mask], full.features.loc[dev_mask], test_start=test_start
    )
    test = TestData(full.symbol, full.bars.loc[~dev_mask], full.features.loc[~dev_mask])
    return dev, test


class WalkForwardConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_folds: int = Field(3, ge=1, le=12)
    mode: Literal["expanding", "rolling"] = "expanding"
    min_train_fraction: float = Field(0.45, gt=0.1, lt=0.95)
    rolling_window: int | None = Field(None, ge=100)
    embargo_bars: int = Field(5, ge=0, le=60)
    horizon: int = Field(1, ge=1, le=20)

    @model_validator(mode="after")
    def _check(self) -> WalkForwardConfig:
        if self.mode == "rolling" and self.rolling_window is None:
            raise ValueError("rolling mode requires rolling_window")
        return self


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold, as positions into the development bars."""

    number: int
    train_positions: np.ndarray = field(repr=False)
    val_start: int
    val_stop: int
    train_start_date: str
    train_end_date: str
    val_start_date: str
    val_end_date: str

    def to_dict(self) -> dict[str, object]:
        return {
            "number": self.number,
            "train_rows": len(self.train_positions),
            "train_start": self.train_start_date,
            "train_end": self.train_end_date,
            "val_start": self.val_start_date,
            "val_end": self.val_end_date,
        }


def usable_positions(frame: MarketFrame) -> np.ndarray:
    """Rows whose full feature vector is defined (after warm-up)."""
    return np.flatnonzero(frame.features.notna().all(axis=1).to_numpy())


def walk_forward_folds(dev: DevelopmentData, config: WalkForwardConfig) -> list[Fold]:
    usable = usable_positions(dev)
    if len(usable) < 200:
        raise ValueError(f"not enough usable development rows ({len(usable)}); need >= 200")
    first, last = int(usable[0]), int(usable[-1]) + 1
    n = last - first
    min_train = int(n * config.min_train_fraction)
    val_len = (n - min_train) // config.n_folds
    if val_len < 40:
        raise ValueError("validation windows would be shorter than 40 sessions")
    gap = config.horizon + 1 + config.embargo_bars  # label span + embargo
    idx = dev.index
    folds = []
    for k in range(config.n_folds):
        val_start = first + min_train + k * val_len
        val_stop = val_start + val_len if k < config.n_folds - 1 else last
        train_hi = val_start - gap  # labels at positions < train_hi end before val_start
        train_lo = first
        if config.mode == "rolling" and config.rolling_window:
            train_lo = max(first, train_hi - config.rolling_window)
        positions = np.arange(train_lo, train_hi)
        if len(positions) < 100:
            raise ValueError("training window shorter than 100 sessions")
        folds.append(
            Fold(
                number=k,
                train_positions=positions,
                val_start=val_start,
                val_stop=val_stop,
                train_start_date=idx[train_lo].date().isoformat(),
                train_end_date=idx[train_hi - 1].date().isoformat(),
                val_start_date=idx[val_start].date().isoformat(),
                val_end_date=idx[val_stop - 1].date().isoformat(),
            )
        )
    return folds


def training_matrix(
    frame: MarketFrame, positions: np.ndarray, *, horizon: int, label_cutoff: int | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Features and labels for ``positions``.

    Labels are computed on bars truncated at ``label_cutoff`` (exclusive), so no
    label can use prices at or beyond it. Rows with undefined features or labels
    are dropped.
    """
    cutoff = len(frame.bars) if label_cutoff is None else label_cutoff
    labels = make_direction_labels(frame.bars.iloc[:cutoff], horizon)
    labels = labels.reindex(frame.index)
    X = frame.features.iloc[positions]
    y = labels.iloc[positions]
    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask], y.loc[mask]
