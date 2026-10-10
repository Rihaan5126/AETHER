"""Dataset import (synthetic, uploaded file, optional Yahoo) and registration."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session
from trading_engine.data import (
    DatasetManifest,
    FileDataProvider,
    SyntheticDataProvider,
    SyntheticMarketSpec,
    YahooFinanceProvider,
    clean_ohlcv,
)

from aether_api.core.config import get_settings
from aether_api.database.models import Dataset, new_id
from aether_api.database.session import session_scope
from aether_api.schemas.datasets import DatasetImportRequest
from aether_api.services.jobs import JobContext, register
from aether_api.services.storage import dataset_store


def register_manifest(db: Session, manifest: DatasetManifest) -> Dataset:
    """Insert a dataset row for ``manifest`` (idempotent on the content hash)."""
    existing = db.scalar(select(Dataset).where(Dataset.hash == manifest.hash))
    if existing is not None:
        return existing
    row = Dataset(
        id=new_id("ds"),
        name=manifest.name,
        source=manifest.source,
        version=manifest.version,
        hash=manifest.hash,
        symbols=manifest.symbols,
        start_date=datetime.fromisoformat(manifest.start_date).date(),
        end_date=datetime.fromisoformat(manifest.end_date).date(),
        rows=manifest.rows,
        params=manifest.params,
        validation=manifest.validation,
        retrieved_at=datetime.fromisoformat(manifest.retrieved_at),
    )
    db.add(row)
    db.flush()
    return row


def import_frame(
    raw: pd.DataFrame, *, name: str, source: str, params: dict[str, Any]
) -> DatasetManifest:
    clean, report = clean_ohlcv(raw)
    return dataset_store().save(
        clean,
        name=name,
        source=source,
        report=report,
        params=params,
        retrieved_at=datetime.now(UTC),
    )


def import_synthetic(request: DatasetImportRequest) -> DatasetManifest:
    spec = SyntheticMarketSpec(**request.synthetic.model_dump())
    raw = SyntheticDataProvider(spec).fetch_ohlcv(
        request.symbols, request.start_date, request.end_date
    )
    return import_frame(
        raw,
        name=request.name,
        source="synthetic",
        params={
            "synthetic": spec.model_dump(),
            "symbols": request.symbols,
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
        },
    )


@register("import_dataset")
def import_dataset_job(ctx: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    request = DatasetImportRequest.model_validate(payload)
    ctx.progress(0.1, f"fetching {request.source} data", force=True)
    if request.source == "synthetic":
        manifest = import_synthetic(request)
    else:
        if not get_settings().enable_yahoo:
            raise RuntimeError("the Yahoo provider is disabled (set AETHER_ENABLE_YAHOO=true)")
        raw = YahooFinanceProvider().fetch_ohlcv(
            request.symbols, request.start_date, request.end_date
        )
        manifest = import_frame(
            raw, name=request.name, source="yahoo", params={"symbols": request.symbols}
        )
    ctx.progress(0.8, "registering dataset version", force=True)
    with session_scope() as db:
        row = register_manifest(db, manifest)
        dataset_id = row.id
    return {
        "dataset_id": dataset_id,
        "hash": manifest.hash,
        "warnings": len([i for i in manifest.validation["issues"] if i["severity"] == "warning"]),
    }


def import_upload(
    data: bytes, filename: str, *, name: str, symbol: str | None, use_adjusted_close: bool
) -> DatasetManifest:
    """Parse an uploaded CSV/Parquet file (size already checked by the route)."""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise ValueError("only .csv and .parquet uploads are accepted")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"  # never trust the client filename for paths
        path.write_bytes(data)
        provider = FileDataProvider(
            path, default_symbol=symbol, use_adjusted_close=use_adjusted_close
        )
        raw = provider.read_all()
    raw["source"] = "upload"
    return import_frame(
        raw,
        name=name,
        source="upload",
        params={"filename": Path(filename).name[:120], "use_adjusted_close": use_adjusted_close},
    )
