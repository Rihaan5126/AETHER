from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthOut(BaseModel):
    status: str
    version: str


class SystemStatus(BaseModel):
    status: str
    version: str
    environment: str
    database: dict[str, Any]
    queue: dict[str, Any]
    jobs: dict[str, int]
    resources: dict[str, Any]
    limits: dict[str, Any]
    storage: dict[str, Any]
    dependencies: dict[str, str]
    code_version: str
    write_protected: bool
    disclaimer: str
