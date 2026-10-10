"""Artifact storage with integrity verification.

Model artifacts live outside the database, under a root directory, one folder per
agent. After saving, every file's SHA-256 is recorded in ``artifact.json`` and in
the database. Loading always re-verifies the hashes first, so a tampered or
foreign file is never deserialised. Only artifacts written by AETHER itself are
ever loaded; there is no API to upload model files.

The interface is small so that an object-storage backend (S3/GCS) can implement
it later; the local backend is the only one shipped.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ArtifactIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactRef:
    key: str
    sha256: str
    files: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {"key": self.key, "sha256": self.sha256, "files": self.files}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_key(*parts: str) -> str:
    for part in parts:
        if not _SAFE_ID.match(part) or ".." in part:
            raise ValueError(f"unsafe artifact key component: {part!r}")
    return "/".join(parts)


class ArtifactStore(ABC):
    @abstractmethod
    def workdir(self, key: str) -> Path:
        """Return an empty local directory to write the artifact for ``key`` into."""

    @abstractmethod
    def commit(self, key: str) -> ArtifactRef:
        """Hash and finalise the artifact written into ``workdir(key)``."""

    @abstractmethod
    def open_verified(self, ref: ArtifactRef) -> Path:
        """Return a local directory for ``ref`` after verifying every file hash."""

    @abstractmethod
    def write_bytes(self, key: str, name: str, data: bytes) -> Path: ...

    @abstractmethod
    def path_for(self, key: str, name: str) -> Path: ...


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents and path != self.root:
            raise ValueError("artifact path escapes the store root")
        return path

    def workdir(self, key: str) -> Path:
        path = self._resolve(key)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        return path

    def commit(self, key: str) -> ArtifactRef:
        path = self._resolve(key)
        files = {
            p.relative_to(path).as_posix(): sha256_file(p)
            for p in sorted(path.rglob("*"))
            if p.is_file() and p.name != "artifact.json"
        }
        digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        (path / "artifact.json").write_text(json.dumps({"sha256": digest, "files": files},
                                                       indent=2))
        return ArtifactRef(key=key, sha256=digest, files=files)

    def open_verified(self, ref: ArtifactRef) -> Path:
        path = self._resolve(ref.key)
        for name, expected in ref.files.items():
            file = (path / name).resolve()
            if path not in file.parents:
                raise ArtifactIntegrityError("artifact file escapes its directory")
            if not file.is_file() or sha256_file(file) != expected:
                raise ArtifactIntegrityError(f"artifact {ref.key}/{name} failed verification")
        return path

    def write_bytes(self, key: str, name: str, data: bytes) -> Path:
        path = self._resolve(key)
        path.mkdir(parents=True, exist_ok=True)
        target = path / safe_key(name)
        target.write_bytes(data)
        return target

    def path_for(self, key: str, name: str) -> Path:
        return self._resolve(key) / safe_key(name)
