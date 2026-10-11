from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from aether_api.schemas.common import APIModel, UTCDateTime

SYMBOL_PATTERN = r"^[A-Za-z][A-Za-z0-9.\-]{0,15}$"


class SyntheticParams(BaseModel):
    seed: int = Field(7, ge=0, le=2**31 - 1)
    annual_drift: float = Field(0.07, ge=-0.5, le=0.5)
    calm_vol: float = Field(0.14, gt=0, le=1.5)
    turbulent_vol: float = Field(0.38, gt=0, le=2.0)
    autocorrelation: float = Field(0.08, ge=-0.5, le=0.5)


class DatasetImportRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    source: Literal["synthetic", "yahoo"] = "synthetic"
    symbols: list[str] = Field(..., min_length=1, max_length=10)
    start_date: date
    end_date: date
    synthetic: SyntheticParams = SyntheticParams()

    @field_validator("symbols")
    @classmethod
    def _symbols(cls, value: list[str]) -> list[str]:
        import re  # noqa: PLC0415

        out = []
        for s in value:
            if not re.match(SYMBOL_PATTERN, s):
                raise ValueError(f"invalid symbol {s!r}")
            out.append(s.upper())
        if len(set(out)) != len(out):
            raise ValueError("duplicate symbols")
        return out

    @model_validator(mode="after")
    def _range(self) -> DatasetImportRequest:
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        if (self.end_date - self.start_date).days > 365 * 40:
            raise ValueError("date range limited to 40 years")
        if (self.end_date - self.start_date).days < 365:
            raise ValueError("at least one year of data is required")
        return self


class DatasetOut(APIModel):
    id: str
    name: str
    source: str
    version: str
    hash: str
    symbols: list[str]
    start_date: date
    end_date: date
    rows: int
    created_at: UTCDateTime
    retrieved_at: UTCDateTime


class DatasetDetail(DatasetOut):
    params: dict[str, Any]
    validation: dict[str, Any]
    used_by_experiments: int = 0


class PricePoint(BaseModel):
    timestamp: date
    open: float
    high: float
    low: float
    close: float
    volume: float


class DatasetPrices(BaseModel):
    dataset_id: str
    symbol: str
    points: list[PricePoint]
    downsampled: bool
