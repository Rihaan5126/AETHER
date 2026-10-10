from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aether_api.core.config import Settings, get_settings
from aether_api.core.errors import APIError, bad_request, not_found
from aether_api.core.security import require_write_access
from aether_api.database.models import Dataset, Experiment
from aether_api.database.session import get_db
from aether_api.routes.deps import Pagination, paginate
from aether_api.schemas.common import JobRef, Page
from aether_api.schemas.datasets import (
    DatasetDetail,
    DatasetImportRequest,
    DatasetOut,
    DatasetPrices,
)
from aether_api.services.datasets import import_upload, register_manifest
from aether_api.services.jobs import create_job, enqueue
from aether_api.services.storage import dataset_store
from trading_engine.data import DataQualityError
from trading_engine.data.schema import symbol_frame

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=Page[DatasetOut])
def list_datasets(page: Pagination = Depends(), source: str | None = None,
                  db: Session = Depends(get_db)) -> Page[DatasetOut]:
    q = select(Dataset).order_by(Dataset.created_at.desc())
    if source:
        q = q.where(Dataset.source == source)
    items, total = paginate(db, q, page)
    return Page[DatasetOut](items=[DatasetOut.model_validate(d) for d in items], total=total,
                            limit=page.limit, offset=page.offset)


@router.post("/import", response_model=JobRef, status_code=status.HTTP_202_ACCEPTED,
             dependencies=[Depends(require_write_access)])
def import_dataset(req: DatasetImportRequest, db: Session = Depends(get_db)) -> JobRef:
    job = create_job(db, "import_dataset", req.model_dump(mode="json"))
    db.commit()
    enqueue(job.id)
    return JobRef(job_id=job.id, status=job.status)


@router.post("/upload", response_model=DatasetOut, status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(require_write_access)])
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(..., min_length=1, max_length=120),
    symbol: str | None = Form(None, pattern=r"^[A-Za-z][A-Za-z0-9.\-]{0,15}$"),
    use_adjusted_close: bool = Form(False),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> DatasetOut:
    limit = settings.max_upload_mb * 2**20
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise APIError(413, "payload_too_large", f"uploads are limited to {settings.max_upload_mb} MB")
    try:
        manifest = import_upload(data, file.filename or "upload.csv", name=name, symbol=symbol,
                                 use_adjusted_close=use_adjusted_close)
    except DataQualityError as exc:
        raise APIError(422, "data_quality", str(exc),
                       exc.report.to_dict() if exc.report else None) from exc
    except (ValueError, KeyError) as exc:
        raise bad_request(f"could not parse upload: {exc}") from exc
    row = register_manifest(db, manifest)
    db.commit()
    return DatasetOut.model_validate(row)


@router.get("/{dataset_id}", response_model=DatasetDetail)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)) -> DatasetDetail:
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise not_found("dataset", dataset_id)
    used = db.scalar(select(func.count()).select_from(Experiment)
                     .where(Experiment.dataset_id == dataset_id)) or 0
    out = DatasetDetail.model_validate(ds)
    out.used_by_experiments = int(used)
    return out


@router.get("/{dataset_id}/prices", response_model=DatasetPrices)
def dataset_prices(dataset_id: str, symbol: str, max_points: int = Query(500, ge=10, le=5000),
                   db: Session = Depends(get_db)) -> DatasetPrices:
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise not_found("dataset", dataset_id)
    if symbol.upper() not in ds.symbols:
        raise bad_request(f"symbol {symbol} not in dataset")
    bars = symbol_frame(dataset_store().load(ds.hash), symbol)
    step = max(1, len(bars) // max_points)
    sampled = bars.iloc[::step]
    points = [{"timestamp": ts.date(), **{k: float(v) for k, v in row.items()}}
              for ts, row in sampled.iterrows()]
    return DatasetPrices(dataset_id=dataset_id, symbol=symbol.upper(), points=points,
                         downsampled=step > 1)
