#!/usr/bin/env bash
set -euo pipefail
export NANOCHAT_BASE_DIR=/workspace/nanochat_cache
mkdir -p logs

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
  echo "EXTRA COMMON ARGS: ${COMMON_ARGS[*]:-<none>}"
  echo "================================================="

  PYTHONUNBUFFERED=1 python -m scripts.base_train "$@" "${COMMON_ARGS[@]}" \
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