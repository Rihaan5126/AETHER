"""Stable experiment manifest (format version 1).

The manifest captures everything needed to reproduce a run: the full config, the
dataset hash, the code revision, interpreter and dependency versions, and the
seed. It is written once when the experiment starts and never modified;
completion data is recorded separately.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from trading_engine.experiments.config import ExperimentConfig

MANIFEST_VERSION = 1
TRACKED_PACKAGES = (
    "trading-engine",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "xgboost",
    "torch",
    "pydantic",
    "optuna",
)


def code_version() -> str:
    env = os.environ.get("AETHER_GIT_COMMIT")
    if env:
        return env
    try:
        root = Path(__file__).resolve().parents[4]
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        if out.returncode != 0 or not sha:
            return "unknown"
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
        return f"{sha}{'-dirty' if dirty.strip() else ''}"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def dependency_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def build_manifest(experiment_id: str, config: ExperimentConfig) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "experiment_id": experiment_id,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_hash": config.dataset_hash,
        "seed": config.seed,
        "code_version": code_version(),
        "dependencies": dependency_versions(),
        "platform": platform.platform(),
        "config": config.to_json_dict(),
    }
