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
  --eval-every=40
  --eval-tokens=524288
  --verbose-log
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

GSO_ARGS=(
  "${BASE_ARGS[@]}"
  --muon-orthogonalization=gso
)

UAGQ_ARGS=(
  "${BASE_ARGS[@]}"
  --muon-orthogonalization=uagq
)

run_method gso_no_norm \
  "${GSO_ARGS[@]}" \
  --model-tag=gso_no_norm \
  --muon-norm-iters=0

run_method gso_frob_norm \
  "${GSO_ARGS[@]}" \
  --model-tag=gso_frob_norm \
  --muon-norm-iters=1 \
  --muon-normalization=frobenius

run_method gso_opt_norm \
  "${GSO_ARGS[@]}" \
  --model-tag=gso_opt_norm \
  --muon-norm-iters=1 \
  --muon-normalization=opt

run_method uagq_no_norm \
  "${UAGQ_ARGS[@]}" \
  --model-tag=uagq_no_norm \
  --muon-norm-iters=0

run_method uagq_frob_norm \
  "${UAGQ_ARGS[@]}" \
  --model-tag=uagq_frob_norm \
  --muon-norm-iters=1 \
  --muon-normalization=frobenius

run_method uagq_opt_norm \
  "${UAGQ_ARGS[@]}" \
  --model-tag=uagq_opt_norm \
  --muon-norm-iters=1 \
  --muon-normalization=opt
