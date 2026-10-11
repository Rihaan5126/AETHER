from __future__ import annotations

from typing import Any

from aether_api.schemas.common import APIModel, UTCDateTime


class JobOut(APIModel):
    id: str
    job_type: str
    status: str
    progress: float
    message: str
    experiment_id: str | None
    attempts: int
    cancel_requested: bool
    result: dict[str, Any] | None
    error_message: str | None
    created_at: UTCDateTime
    started_at: UTCDateTime | None
    completed_at: UTCDateTime | None
    heartbeat_at: UTCDateTime | None
