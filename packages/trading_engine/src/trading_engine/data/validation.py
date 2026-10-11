"""Data quality validation and explicit cleaning rules.

Rules (documented in docs/methodology.md, "Data handling"):

==========================  ========  ==============================================
check                       severity  action in :func:`clean_ohlcv`
==========================  ========  ==============================================
missing required column     error     raise immediately
non-finite / missing price  error     drop row (treated as a missing observation)
non-positive price          error     drop row
OHLC relationship broken    error     drop row
negative volume             error     drop row
exact duplicate row         warning   keep first
conflicting duplicate       error     drop *all* rows for that (symbol, date)
weekend session             warning   keep (reported)
calendar misalignment       warning   keep; simulator treats gaps as missing data
long gap (> 5 sessions)     warning   keep; possible halt or data outage
series ends early           warning   keep; possible delisting (survivorship!)
|daily return| > 40%        warning   keep; possible unadjusted split
==========================  ========  ==============================================

If dropped rows exceed ``max_invalid_fraction`` of the input, cleaning fails with
:class:`DataQualityError` instead of silently producing a thin dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from trading_engine.data.schema import ADJUSTMENT_VALUES, OHLCV_COLUMNS, PRICE_COLUMNS

Severity = Literal["error", "warning"]


class DataQualityError(ValueError):
    """Raised when a dataset fails validation beyond the configured tolerance."""

    def __init__(self, message: str, report: ValidationReport | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    symbol: str | None = None
    count: int = 1


@dataclass
class ValidationReport:
    rows_in: int
    rows_out: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    def to_dict(self) -> dict[str, object]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "issues": [issue.__dict__ for issue in self.issues],
        }


def _row_problems(frame: pd.DataFrame) -> dict[str, pd.Series]:
    prices = frame.loc[:, list(PRICE_COLUMNS)]
    nonfinite = ~np.isfinite(prices.to_numpy(dtype="float64")).all(axis=1)
    nonfinite_s = pd.Series(nonfinite, index=frame.index)
    nonpositive = (prices <= 0).any(axis=1) & ~nonfinite_s
    upper = prices[["open", "close", "low"]].max(axis=1)
    lower = prices[["open", "close"]].min(axis=1)
    tol = 1e-9
    broken = ((frame["high"] < upper - tol) | (frame["low"] > lower + tol)) & ~nonfinite_s
    vol = pd.to_numeric(frame["volume"], errors="coerce")
    bad_volume = vol.isna() | (vol < 0)
    return {
        "missing_price": nonfinite_s,
        "non_positive_price": nonpositive,
        "ohlc_relationship": broken,
        "invalid_volume": bad_volume,
    }


def validate_ohlcv(frame: pd.DataFrame) -> ValidationReport:
    """Inspect ``frame`` without modifying it."""
    _, report = _clean(frame, max_invalid_fraction=1.0, raise_on_failure=False)
    return report


def clean_ohlcv(
    frame: pd.DataFrame, *, max_invalid_fraction: float = 0.01
) -> tuple[pd.DataFrame, ValidationReport]:
    """Apply the cleaning rules and return ``(clean_frame, report)``."""
    return _clean(frame, max_invalid_fraction=max_invalid_fraction, raise_on_failure=True)


def _clean(
    frame: pd.DataFrame, *, max_invalid_fraction: float, raise_on_failure: bool
) -> tuple[pd.DataFrame, ValidationReport]:
    report = ValidationReport(rows_in=len(frame))
    missing_cols = set(OHLCV_COLUMNS) - set(frame.columns)
    if missing_cols:
        report.add(
            ValidationIssue("error", "missing_columns", f"missing columns {sorted(missing_cols)}")
        )
        raise DataQualityError(f"missing columns {sorted(missing_cols)}", report)
    if frame.empty:
        report.add(ValidationIssue("error", "empty", "dataset has no rows"))
        raise DataQualityError("dataset has no rows", report)

    work = frame.loc[:, list(OHLCV_COLUMNS)].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"]).dt.normalize()
    for col in (*PRICE_COLUMNS, "volume"):
        work[col] = pd.to_numeric(work[col], errors="coerce").astype("float64")

    bad_adj = ~work["adjustment"].isin(ADJUSTMENT_VALUES)
    if bad_adj.any():
        report.add(
            ValidationIssue(
                "error", "invalid_adjustment", "unknown adjustment label", None, int(bad_adj.sum())
            )
        )

    drop = pd.Series(False, index=work.index)
    for code, mask in _row_problems(work).items():
        for symbol, count in work.loc[mask, "symbol"].value_counts().items():
            report.add(
                ValidationIssue("error", code, f"{count} rows with {code}", str(symbol), int(count))
            )
        drop |= mask
    drop |= bad_adj

    key = ["symbol", "timestamp"]
    dup_all = work.duplicated(key, keep=False)
    if dup_all.any():
        exact_dup = work.duplicated(keep="first")
        if exact_dup.any():
            report.add(
                ValidationIssue(
                    "warning",
                    "exact_duplicate",
                    "exact duplicate rows removed",
                    None,
                    int(exact_dup.sum()),
                )
            )
        drop |= exact_dup
        remaining = work.loc[~exact_dup]
        conflict_keys = remaining.loc[remaining.duplicated(key, keep=False), key]
        if not conflict_keys.empty:
            conflict = work.set_index(key).index.isin(
                pd.MultiIndex.from_frame(conflict_keys.drop_duplicates())
            )
            report.add(
                ValidationIssue(
                    "error",
                    "conflicting_duplicate",
                    "conflicting rows for the same session were dropped",
                    None,
                    int(conflict.sum()),
                )
            )
            drop |= pd.Series(conflict, index=work.index)

    clean = work.loc[~drop].sort_values(key, kind="mergesort").reset_index(drop=True)
    report.rows_out = len(clean)
    _calendar_checks(clean, report)

    dropped_fraction = 1 - report.rows_out / max(report.rows_in, 1)
    if raise_on_failure and (dropped_fraction > max_invalid_fraction or clean.empty):
        raise DataQualityError(
            f"{dropped_fraction:.2%} of rows failed validation "
            f"(tolerance {max_invalid_fraction:.2%})",
            report,
        )
    return clean, report


def _calendar_checks(clean: pd.DataFrame, report: ValidationReport) -> None:
    if clean.empty:
        return
    weekend = clean["timestamp"].dt.dayofweek >= 5
    if weekend.any():
        report.add(
            ValidationIssue(
                "warning", "weekend_session", "rows dated on a weekend", None, int(weekend.sum())
            )
        )
    calendar = pd.DatetimeIndex(clean["timestamp"].unique()).sort_values()
    global_end = calendar.max()
    for symbol, sub in clean.groupby("symbol", sort=True):
        dates = pd.DatetimeIndex(sub["timestamp"])
        span = calendar[(calendar >= dates.min()) & (calendar <= dates.max())]
        missing = span.difference(dates)
        if len(missing):
            report.add(
                ValidationIssue(
                    "warning",
                    "calendar_misalignment",
                    f"{len(missing)} sessions present for other symbols are missing",
                    str(symbol),
                    len(missing),
                )
            )
        bdays = np.busday_count(
            dates[:-1].values.astype("datetime64[D]"), dates[1:].values.astype("datetime64[D]")
        )
        long_gaps = int((bdays > 5).sum())
        if long_gaps:
            report.add(
                ValidationIssue(
                    "warning",
                    "long_gap",
                    "gaps longer than 5 sessions (halt or outage?)",
                    str(symbol),
                    long_gaps,
                )
            )
        if dates.max() < global_end - pd.Timedelta(days=10):
            report.add(
                ValidationIssue(
                    "warning",
                    "ends_early",
                    f"series ends {dates.max().date()} before dataset end (possible delisting)",
                    str(symbol),
                )
            )
        rets = sub["close"].pct_change().abs()
        jumps = int((rets > 0.4).sum())
        if jumps:
            report.add(
                ValidationIssue(
                    "warning",
                    "extreme_return",
                    "daily moves above 40% (unadjusted split?)",
                    str(symbol),
                    jumps,
                )
            )
