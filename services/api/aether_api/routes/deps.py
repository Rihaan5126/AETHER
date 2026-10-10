"""Shared route dependencies and helpers."""

from __future__ import annotations

from typing import Any, TypeVar

from fastapi import Query
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

T = TypeVar("T")


class Pagination:
    def __init__(self, limit: int = Query(50, ge=1, le=200),
                 offset: int = Query(0, ge=0, le=1_000_000)) -> None:
        self.limit = limit
        self.offset = offset


def paginate(db: Session, query: Select[Any], page: Pagination) -> tuple[list[Any], int]:
    total = db.scalar(select(func.count()).select_from(query.order_by(None).subquery())) or 0
    items = list(db.scalars(query.limit(page.limit).offset(page.offset)))
    return items, int(total)
