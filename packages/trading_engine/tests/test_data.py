from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from trading_engine.data import (
    DataQualityError,
    DatasetStore,
    FileDataProvider,
    SyntheticDataProvider,
    SyntheticMarketSpec,
    clean_ohlcv,
    compute_dataset_hash,
    validate_ohlcv,
)


def codes(report):
    return {i.code for i in report.issues}


def test_synthetic_is_deterministic_and_seed_sensitive():
    a = SyntheticDataProvider().fetch_ohlcv(["AAA", "BBB"], date(2020, 1, 1), date(2021, 1, 1))
    b = SyntheticDataProvider().fetch_ohlcv(["AAA", "BBB"], date(2020, 1, 1), date(2021, 1, 1))
    c = SyntheticDataProvider(SyntheticMarketSpec(seed=8)).fetch_ohlcv(
        ["AAA", "BBB"], date(2020, 1, 1), date(2021, 1, 1)
    )
    assert compute_dataset_hash(a) == compute_dataset_hash(b)
    assert compute_dataset_hash(a) != compute_dataset_hash(c)


def test_synthetic_bars_pass_validation(synthetic_dataset):
    clean, report = clean_ohlcv(synthetic_dataset)
    assert report.errors == []
    assert len(clean) == len(synthetic_dataset)
    assert (clean["high"] >= clean[["open", "close"]].max(axis=1)).all()
    assert (clean["low"] <= clean[["open", "close"]].min(axis=1)).all()


def test_validation_detects_and_drops_bad_rows(synthetic_dataset):
    bad = synthetic_dataset.head(400).copy()
    bad.loc[3, "close"] = np.nan  # missing observation
    bad.loc[5, "high"] = bad.loc[5, "low"] * 0.5  # broken OHLC relationship
    bad.loc[7, "volume"] = -1  # negative volume
    bad.loc[9, "open"] = -2.0  # non-positive price
    dup = bad.loc[[20]].copy()  # exact duplicate
    conflict = bad.loc[[30]].copy()
    conflict["close"] *= 1.01  # conflicting duplicate
    bad = pd.concat([bad, dup, conflict], ignore_index=True)
    report = validate_ohlcv(bad)
    assert {
        "missing_price",
        "ohlc_relationship",
        "invalid_volume",
        "non_positive_price",
        "exact_duplicate",
        "conflicting_duplicate",
    } <= codes(report)
    clean, _ = clean_ohlcv(bad, max_invalid_fraction=0.05)
    assert not clean.duplicated(["symbol", "timestamp"]).any()
    # 4 invalid rows + both rows of the conflicting session are removed
    assert len(clean) == 400 - 4 - 1


def test_cleaning_fails_beyond_tolerance(synthetic_dataset):
    bad = synthetic_dataset.head(100).copy()
    bad.loc[:10, "close"] = np.nan
    with pytest.raises(DataQualityError):
        clean_ohlcv(bad, max_invalid_fraction=0.01)


def test_calendar_misalignment_and_gap_warnings():
    data = SyntheticDataProvider().fetch_ohlcv(["AAA", "BBB"], date(2020, 1, 1), date(2020, 12, 31))
    holes = data[
        ~((data["symbol"] == "BBB") & data["timestamp"].between("2020-03-02", "2020-03-20"))
    ]
    report = validate_ohlcv(holes)
    assert "calendar_misalignment" in codes(report)
    assert "long_gap" in codes(report)


def test_dataset_store_is_immutable_and_verifies_integrity(tmp_path, synthetic_dataset):
    store = DatasetStore(tmp_path)
    clean, report = clean_ohlcv(synthetic_dataset)
    m1 = store.save(clean, name="syn", source="synthetic", report=report)
    m2 = store.save(clean.sample(frac=1.0, random_state=1), name="other", source="x", report=report)
    assert m1.hash == m2.hash and m2.name == "syn"  # same content -> same version, not replaced
    loaded = store.load(m1.hash)
    assert compute_dataset_hash(loaded) == m1.hash
    # Tampering is detected.
    path = tmp_path / m1.hash / "data.parquet"
    tampered = loaded.copy()
    tampered.loc[0, "close"] += 1
    tampered.to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="integrity"):
        store.load(m1.hash)


def test_file_provider_roundtrip_and_adjustment(tmp_path):
    df = pd.DataFrame(
        {
            "Date": pd.bdate_range("2021-01-04", periods=5),
            "Open": [10, 11, 12, 13, 14.0],
            "High": [11, 12, 13, 14, 15.0],
            "Low": [9, 10, 11, 12, 13.0],
            "Close": [10.5, 11.5, 12.5, 13.5, 14.5],
            "Adj Close": [5.25, 5.75, 6.25, 6.75, 7.25],
            "Volume": [100, 200, 300, 400, 500],
        }
    )
    path = tmp_path / "x.csv"
    df.to_csv(path, index=False)
    raw = FileDataProvider(path, default_symbol="xyz").read_all()
    assert raw["symbol"].unique().tolist() == ["XYZ"] and raw["adjustment"].iloc[0] == "none"
    adj = FileDataProvider(path, default_symbol="xyz", use_adjusted_close=True).read_all()
    assert np.allclose(adj["close"], df["Adj Close"]) and np.allclose(adj["open"], df["Open"] / 2)
    assert adj["adjustment"].iloc[0] == "split_dividend"


def test_file_provider_rejects_unknown_types(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="unsupported"):
        FileDataProvider(p, default_symbol="A").read_all()
