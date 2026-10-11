"""Immutable, content-addressed dataset versions.

A dataset version is identified by the SHA-256 of a canonical serialisation of its
rows (sorted, fixed column order, fixed float formatting). The same bytes always
produce the same hash, and a stored version is never overwritten: saving a frame
whose hash already exists returns the existing manifest unchanged. Experiments
reference datasets by this hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from trading_engine.data.schema import OHLCV_COLUMNS, sort_canonical
from trading_engine.data.validation import ValidationReport


def canonical_bytes(frame: pd.DataFrame) -> bytes:
    canon = sort_canonical(frame)
    canon["timestamp"] = canon["timestamp"].dt.strftime("%Y-%m-%d")
    return canon.to_csv(index=False, float_format="%.6f", lineterminator="\n").encode("utf-8")


def compute_dataset_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(canonical_bytes(frame)).hexdigest()


class DatasetManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    hash: str
    name: str
    source: str
    symbols: list[str]
    start_date: str
    end_date: str
    rows: int
    adjustment: list[str]
    retrieved_at: str
    created_at: str
    validation: dict[str, Any]
    params: dict[str, Any] = {}

    @property
    def version(self) -> str:
        return self.hash[:12]


class DatasetStore:
    """Filesystem store: ``<root>/<hash>/data.parquet`` + ``manifest.json``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, dataset_hash: str) -> Path:
        if len(dataset_hash) != 64 or any(c not in "0123456789abcdef" for c in dataset_hash):
            raise ValueError("invalid dataset hash")
        return self.root / dataset_hash

    def exists(self, dataset_hash: str) -> bool:
        return (self._dir(dataset_hash) / "manifest.json").is_file()

    def save(
        self,
        frame: pd.DataFrame,
        *,
        name: str,
        source: str,
        report: ValidationReport,
        retrieved_at: datetime | None = None,
        params: dict[str, Any] | None = None,
    ) -> DatasetManifest:
        canon = sort_canonical(frame)
        digest = compute_dataset_hash(canon)
        if self.exists(digest):
            return self.manifest(digest)
        target = self._dir(digest)
        target.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC).isoformat()
        manifest = DatasetManifest(
            hash=digest,
            name=name,
            source=source,
            symbols=sorted(canon["symbol"].unique().tolist()),
            start_date=canon["timestamp"].min().date().isoformat(),
            end_date=canon["timestamp"].max().date().isoformat(),
            rows=len(canon),
            adjustment=sorted(canon["adjustment"].unique().tolist()),
            retrieved_at=(retrieved_at or datetime.now(UTC)).isoformat(),
            created_at=now,
            validation=report.to_dict(),
            params=params or {},
        )
        canon.to_parquet(target / "data.parquet", index=False)
        _atomic_write(target / "manifest.json", manifest.model_dump_json(indent=2))
        return manifest

    def manifest(self, dataset_hash: str) -> DatasetManifest:
        path = self._dir(dataset_hash) / "manifest.json"
        return DatasetManifest.model_validate(json.loads(path.read_text()))

    def load(self, dataset_hash: str, *, verify: bool = True) -> pd.DataFrame:
        frame = pd.read_parquet(self._dir(dataset_hash) / "data.parquet")
        frame = frame.loc[:, list(OHLCV_COLUMNS)]
        if verify and compute_dataset_hash(frame) != dataset_hash:
            raise RuntimeError(f"dataset {dataset_hash[:12]} failed integrity check")
        return frame

    def list(self) -> list[DatasetManifest]:
        out = []
        for child in sorted(self.root.iterdir()):
            if (child / "manifest.json").is_file():
                out.append(self.manifest(child.name))
        return out

    def cache_path(self, dataset_hash: str, key: str) -> Path:
        """Path for derived, re-computable artefacts (e.g. feature matrices)."""
        safe = "".join(c for c in key if c.isalnum() or c in "-_")
        cache = self._dir(dataset_hash) / "cache"
        cache.mkdir(exist_ok=True)
        return cache / f"{safe}.parquet"


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
    os.replace(tmp, path)
