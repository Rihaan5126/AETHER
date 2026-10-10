# syntax=docker/dockerfile:1.7
# Image for both the API and the Celery training worker (different commands).
ARG PYTHON_IMAGE=python:3.13-slim
FROM ${PYTHON_IMAGE}
# CPU-only PyTorch keeps the image ~3 GB smaller than the default CUDA wheels.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG GIT_COMMIT=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    AETHER_GIT_COMMIT=${GIT_COMMIT}

# No system packages needed: numpy, scikit-learn and XGBoost wheels vendor OpenMP.
RUN --mount=type=secret,id=ca \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install --no-cache-dir uv==0.11.32

WORKDIR /app
# Dependencies first for layer caching: everything pinned in uv.lock, with torch
# from the CPU wheel index (see scripts/install-cpu.sh). An optional BuildKit
# secret "ca" lets builds behind a TLS-intercepting proxy trust its certificate.
COPY pyproject.toml uv.lock ./
COPY packages/trading_engine/pyproject.toml packages/trading_engine/
COPY services/api/pyproject.toml services/api/
COPY scripts/install-cpu.sh scripts/
RUN --mount=type=cache,target=/root/.cache/uv --mount=type=secret,id=ca \
    if [ -f /run/secrets/ca ]; then export SSL_CERT_FILE=/run/secrets/ca; fi; \
    TORCH_INDEX_URL="${TORCH_INDEX_URL}" scripts/install-cpu.sh /opt/venv

COPY packages packages
COPY services services
RUN --mount=type=cache,target=/root/.cache/uv --mount=type=secret,id=ca \
    if [ -f /run/secrets/ca ]; then export SSL_CERT_FILE=/run/secrets/ca; fi; \
    uv pip install --python /opt/venv/bin/python --no-deps ./packages/trading_engine ./services/api

RUN useradd --create-home --uid 10001 aether \
    && mkdir -p /data/datasets /data/artifacts && chown -R aether /data
USER aether
ENV AETHER_DATA_DIR=/data/datasets \
    AETHER_ARTIFACT_DIR=/data/artifacts

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1
CMD ["sh", "-c", "alembic -c services/api/alembic.ini upgrade head && uvicorn aether_api.main:app --host 0.0.0.0 --port 8000"]
