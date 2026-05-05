#!/usr/bin/env bash
set -euo pipefail

cd /workspace/nanochat_apo

export NANOCHAT_BASE_DIR=/workspace/nanochat_cache
export TMPDIR=/workspace/pip_tmp
export PIP_CACHE_DIR=/workspace/pip_cache

PYTHON_BIN=/workspace/nanochat_apo/.venv/bin/python

mkdir -p logs "$TMPDIR" "$PIP_CACHE_DIR"

"$PYTHON_BIN" - <<'PY'
import torch
from nanochat.common import get_base_dir
from nanochat.dataset import list_parquet_files

paths = list_parquet_files()
print("Python/Torch OK")
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("base_dir:", get_base_dir())
print("dataset files:", len(paths))
print("train files:", len(paths)-1)
print("val:", paths[-1])
PY

COMMON_ARGS=(
  --muon-orthogonalization-dtype=float32
  --eval-every=10
  --eval-tokens=524288
  --num-iterations=100
  --verbose-log
)

run_method () {
  local name="$1"
  shift

  echo "================================================="
  echo "STARTING: $name"
  echo "TIME: $(date)"
  echo "PYTHON: $PYTHON_BIN"
  echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
  echo "EXTRA COMMON ARGS: ${COMMON_ARGS[*]:-<none>}"
  echo "METHOD ARGS: $*"
  echo "================================================="

  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m scripts.base_train "$@" "${COMMON_ARGS[@]}" \
    2>&1 | tee "logs/${name}.log"

  echo "================================================="
  echo "FINISHED: $name"
  echo "TIME: $(date)"
  echo "================================================="
}

BASE_ARGS=(
  --core-metric-every=-1
  --depth=14
  --window-pattern=L
  --sample-every=-1
  --save-every=-1
  --run=dummy
  --ns-steps=5
  --print-every=1
)

LRS=(0.01 0.02 0.025 0.03 0.04)

for LR in "${LRS[@]}"; do
  LR_TAG="${LR//./p}"

  run_method "polar_express_lr${LR_TAG}" \
    "${BASE_ARGS[@]}" \
    --muon-orthogonalization=polar_express \
    --model-tag="polar_express_lr${LR_TAG}" \
    --matrix-lr="${LR}"

  run_method "gso_lr${LR_TAG}" \
    "${BASE_ARGS[@]}" \
    --muon-orthogonalization=gso \
    --model-tag="gso_lr${LR_TAG}" \
    --muon-norm-iters=1 \
    --matrix-lr="${LR}"

  run_method "muon_adhoc_lr${LR_TAG}" \
    "${BASE_ARGS[@]}" \
    --muon-orthogonalization=muon_adhoc \
    --model-tag="muon_adhoc_lr${LR_TAG}" \
    --ortho-order=1 \
    --muon-norm-iters=1 \
    --matrix-lr="${LR}"

  run_method "newton_schulz_2_lr${LR_TAG}" \
    "${BASE_ARGS[@]}" \
    --muon-orthogonalization=newton_schulz \
    --model-tag="newton_schulz_2_lr${LR_TAG}" \
    --ortho-order=2 \
    --muon-norm-iters=1 \
    --matrix-lr="${LR}"

done