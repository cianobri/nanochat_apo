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
  --muon-orthogonalization-dtype=bfloat16
  --eval-every=40
  --eval-tokens=524288
)

if [[ -n "${NUM_ITERATIONS:-}" ]]; then
  COMMON_ARGS+=(--num-iterations="${NUM_ITERATIONS}")
fi

run_method () {
  local name="$1"
  shift

  echo "================================================="
  echo "STARTING: $name"
  echo "TIME: $(date)"
  echo "PYTHON: $PYTHON_BIN"
  echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
  echo "EXTRA COMMON ARGS: ${COMMON_ARGS[*]:-<none>}"
  echo "================================================="

  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m scripts.base_train "$@" "${COMMON_ARGS[@]}" \
    2>&1 | tee "logs/${name}.log"

  echo "================================================="
  echo "FINISHED: $name"
  echo "TIME: $(date)"
  echo "================================================="
}

run_method newton_schulz_2 \
  --core-metric-every=-1 \
  --depth=14 \
  --window-pattern=L \
  --sample-every=-1 \
  --save-every=-1 \
  --run=dummy \
  --muon-orthogonalization=newton_schulz \
  --model-tag=newton_schulz_2 \
  --ortho-order=2 \
  --ns-steps=5 \
  --muon-norm-iters=1 \
  --print-every=1

run_method gso \
  --core-metric-every=-1 \
  --depth=14 \
  --window-pattern=L \
  --sample-every=-1 \
  --save-every=-1 \
  --run=dummy \
  --muon-orthogonalization=gso \
  --model-tag=gso \
  --ns-steps=5 \
  --muon-norm-iters=1 \
  --print-every=1

run_method polar_express \
  --core-metric-every=-1 \
  --depth=14 \
  --window-pattern=L \
  --sample-every=-1 \
  --save-every=-1 \
  --run=dummy \
  --muon-orthogonalization=polar_express \
  --model-tag=polar_express \
  --ns-steps=5 \
  --print-every=1

run_method muon_adhoc \
  --core-metric-every=-1 \
  --depth=14 \
  --window-pattern=L \
  --sample-every=-1 \
  --save-every=-1 \
  --run=dummy \
  --muon-orthogonalization=muon_adhoc \
  --model-tag=muon_adhoc \
  --ortho-order=1 \
  --ns-steps=5 \
  --muon-norm-iters=1 \
  --print-every=1

run_method ls2 \
  --core-metric-every=-1 \
  --depth=14 \
  --window-pattern=L \
  --sample-every=-1 \
  --save-every=-1 \
  --run=dummy \
  --muon-orthogonalization=ls2 \
  --model-tag=ls2 \
  --ns-steps=5 \
  --muon-norm-iters=1 \
  --print-every=1

run_method adamW \
  --optimizer=adamw \
  --core-metric-every=-1 \
  --depth=14 \
  --window-pattern=L \
  --sample-every=-1 \
  --save-every=-1 \
  --run=dummy \
  --model-tag=adamW \
  --print-every=1