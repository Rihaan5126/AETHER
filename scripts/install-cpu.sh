#!/usr/bin/env bash
# Install the locked Python dependencies with CPU-only PyTorch.
#
# `uv sync` installs the default PyPI torch wheel, which on Linux pulls ~3 GB of
# CUDA libraries. AETHER's MLP agents are tiny and train on CPU, so Docker and CI
# install every other dependency exactly as pinned in uv.lock, then the same
# torch version from the CPU wheel index.
#
# Usage: scripts/install-cpu.sh <venv-dir> [--dev]
set -euo pipefail
VENV="${1:?usage: install-cpu.sh <venv-dir> [--dev]}"
DEV_FLAG="--no-dev"
[[ "${2:-}" == "--dev" ]] && DEV_FLAG=""
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cd "$ROOT"
uv export --frozen --no-hashes --no-emit-workspace $DEV_FLAG -o "$TMP/all.txt" >/dev/null
TORCH_VERSION="$(sed -nE 's/^torch==([^ ;]+).*/\1/p' "$TMP/all.txt" | head -1)"
grep -vE '^(torch|triton|nvidia-|cuda-)' "$TMP/all.txt" > "$TMP/no-torch.txt"

[[ -x "$VENV/bin/python" ]] || uv venv --python 3.13 "$VENV"
uv pip install --python "$VENV/bin/python" -r "$TMP/no-torch.txt"
uv pip install --python "$VENV/bin/python" "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX_URL"
echo "Installed torch ${TORCH_VERSION} from ${TORCH_INDEX_URL}"
