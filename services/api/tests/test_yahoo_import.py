"""Yahoo import path, exercised offline: the network download is replaced by synthetic bars."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd
import pytest
from aether_api import cli
from aether_api.services import datasets as dataset_service
from conftest import wait_for_job
from trading_engine.data import SyntheticDataProvider, SyntheticMarketSpec


def _fake_fetch(self: object, symbols: Sequence[str], start: date, end: date) -> pd.DataFrame:
    frame = SyntheticDataProvider(SyntheticMarketSpec(seed=11)).fetch_ohlcv(symbols, start, end)
    frame["adjustment"] = "split_dividend"
    frame["source"] = "yahoo"
    return frame


@pytest.fixture
def offline_yahoo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dataset_service.YahooFinanceProvider, "fetch_ohlcv", _fake_fetch)


def test_cli_import_registers_versioned_yahoo_dataset(offline_yahoo: None) -> None:
    result = cli.import_yahoo_dataset(["spy"], date(2015, 1, 1), date(2018, 12, 31), None)
    assert result["symbols"] == ["SPY"]
    assert result["rows"] and int(str(result["rows"])) > 900

    listed = {d["id"]: d for d in cli.list_datasets()}
    assert listed[str(result["dataset_id"])]["source"] == "yahoo"

    # Identical bars hash to the same immutable version instead of a duplicate.
    again = cli.import_yahoo_dataset(["SPY"], date(2015, 1, 1), date(2018, 12, 31), None)
    assert again["dataset_id"] == result["dataset_id"]


def test_import_records_requested_range_and_adjustment(offline_yahoo: None) -> None:
    from aether_api.schemas.datasets import DatasetImportRequest

    manifest = dataset_service.import_yahoo(
        DatasetImportRequest(
            name="offline yahoo",
            source="yahoo",
            symbols=["QQQ"],
            start_date=date(2016, 1, 1),
            end_date=date(2019, 6, 30),
        )
    )
    assert manifest.source == "yahoo"
    assert manifest.params["start_date"] == "2016-01-01"
    assert "auto_adjust" in manifest.params["adjustment"]


def test_api_yahoo_import_is_disabled_by_default(client, offline_yahoo: None) -> None:
    r = client.post(
        "/api/v1/datasets/import",
        json={
            "name": "should not import",
            "source": "yahoo",
            "symbols": ["SPY"],
            "start_date": "2015-01-01",
            "end_date": "2018-12-31",
        },
    )
    assert r.status_code == 202, r.text
    job = wait_for_job(client, r.json()["job_id"])
    assert job["status"] == "failed"
    assert "AETHER_ENABLE_YAHOO" in (job["error_message"] or "")
