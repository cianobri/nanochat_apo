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
  --core-metric-every=-1
  --depth=12
  --window-pattern=L
  --sample-every=-1
  --save-every=-1
  --run=dummy
  --ns-steps=5
  --print-every=10
)

# Optional override:
#   NUM_ITERATIONS=500 ./run_norm_ablation.sh
if [[ -n "${NUM_ITERATIONS:-}" ]]; then
  COMMON_ARGS+=(--num-iterations="${NUM_ITERATIONS}")
fi

run_method () {
  local name="$1"
  shift

  echo "================================================="
  echo "STARTING: ${name}"
  echo "TIME: $(date)"
  echo "PYTHON: $PYTHON_BIN"
  echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
  echo "COMMON_ARGS: ${COMMON_ARGS[*]}"
  echo "METHOD_ARGS: $*"
  echo "================================================="

  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m scripts.base_train \
    "${COMMON_ARGS[@]}" \
    "$@" \
    2>&1 | tee "logs/${name}.log"

  echo "================================================="
  echo "FINISHED: ${name}"
  echo "TIME: $(date)"
  echo "================================================="
}

# -------------------------------------------------
# Polar Express, normalized and raw
# -------------------------------------------------

run_method pe_norm \
  --muon-orthogonalization=polar_express \
  --model-tag=pe_norm \
  --muon-norm-iters=1

run_method pe_raw \
  --muon-orthogonalization=polar_express \
  --model-tag=pe_raw \
  --muon-norm-iters=0

# -------------------------------------------------
# Newton-Schulz, normalized and raw
# -------------------------------------------------

run_method ns_norm \
  --muon-orthogonalization=newton_schulz \
  --model-tag=ns_norm \
  --muon-norm-iters=1

run_method ns_raw \
  --muon-orthogonalization=newton_schulz \
  --model-tag=ns_raw \
  --muon-norm-iters=0

# -------------------------------------------------
# FO-APO, normalized and raw
# -------------------------------------------------

run_method apo1_norm \
  --muon-orthogonalization=adaptive_poly \
  --model-tag=apo1_norm \
  --ortho-order=1 \
  --muon-norm-iters=1

run_method apo1_raw \
  --muon-orthogonalization=adaptive_poly \
  --model-tag=apo1_raw \
  --ortho-order=1 \
  --muon-norm-iters=0

# -------------------------------------------------
# SO-APO, normalized and raw
# -------------------------------------------------

run_method apo2_norm \
  --muon-orthogonalization=adaptive_poly \
  --model-tag=apo2_norm \
  --ortho-order=2 \
  --muon-norm-iters=1

run_method apo2_raw \
  --muon-orthogonalization=adaptive_poly \
  --model-tag=apo2_raw \
  --ortho-order=2 \
  --muon-norm-iters=0