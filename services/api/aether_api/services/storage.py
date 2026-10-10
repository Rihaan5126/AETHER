"""Access to the dataset store, artifact store and Parquet time-series artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from trading_engine.data import DatasetStore
from trading_engine.storage import LocalArtifactStore, safe_key, sha256_file

from aether_api.core.config import get_settings


def dataset_store() -> DatasetStore:
    return DatasetStore(get_settings().data_dir)


def artifact_store() -> LocalArtifactStore:
    return LocalArtifactStore(get_settings().artifact_dir)


def write_parquet_artifact(frame: pd.DataFrame, *parts: str) -> tuple[str, str]:
    """Write ``frame`` under the artifact root; returns (relative path, sha256)."""
    store = artifact_store()
    key = safe_key(*parts[:-1])
    path = store.path_for(key, parts[-1])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    return f"{key}/{parts[-1]}", sha256_file(path)


def read_parquet_artifact(relative: str, expected_sha256: str | None = None) -> pd.DataFrame:
    root = artifact_store().root
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("artifact path escapes the store root")
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise RuntimeError(f"artifact {relative} failed integrity check")
    return pd.read_parquet(path)


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def ensure_dirs() -> None:
    settings = get_settings()
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.artifact_dir).mkdir(parents=True, exist_ok=True)
