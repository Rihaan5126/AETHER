"""Write protection and rate limiting for mutating endpoints.

* If ``AETHER_API_KEY`` is set, every POST/DELETE requires ``X-API-Key``.
  In production the key is mandatory (the app refuses to start without it), so a
  public deployment can never expose an unauthenticated training trigger.
* A per-client sliding-window limiter caps write requests per minute. It is
  in-process; deployments with several API replicas should front it with a
  shared limiter (documented in docs/architecture.md).
"""

from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, Header, Request

from aether_api.core.config import Settings, get_settings
from aether_api.core.errors import APIError


class RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window: float = 60.0) -> None:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > window:
                hits.popleft()
            if len(hits) >= limit:
                raise APIError(429, "rate_limited", "too many write requests; slow down")
            hits.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


rate_limiter = RateLimiter()


def require_write_access(
    request: Request,
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if settings.api_key and (not x_api_key or not hmac.compare_digest(x_api_key, settings.api_key)):
        raise APIError(401, "unauthorized", "missing or invalid X-API-Key")
    client = request.client.host if request.client else "unknown"
    rate_limiter.check(client, settings.write_rate_limit_per_minute)
