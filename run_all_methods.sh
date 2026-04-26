#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

COMMON_ARGS=()

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
  --print-every=50

run_method newton_schulz_1 \
  --core-metric-every=-1 \
  --depth=14 \
  --window-pattern=L \
  --sample-every=-1 \
  --save-every=-1 \
  --run=dummy \
  --muon-orthogonalization=newton_schulz \
  --model-tag=newton_schulz_1 \
  --ortho-order=1 \
  --ns-steps=5 \
  --muon-norm-iters=1 \
  --print-every=50

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
  --print-every=50

run_method adaptive_poly_1 \
  --core-metric-every=-1 \
  --depth=14 \
  --window-pattern=L \
  --sample-every=-1 \
  --save-every=-1 \
  --run=dummy \
  --muon-orthogonalization=adaptive_poly \
  --model-tag=adaptive_poly_1 \
  --ortho-order=1 \
  --ns-steps=2 \
  --muon-norm-iters=0 \
  --print-every=50

run_method adaptive_poly_2 \
  --core-metric-every=-1 \
  --depth=14 \
  --window-pattern=L \
  --sample-every=-1 \
  --save-every=-1 \
  --run=dummy \
  --muon-orthogonalization=adaptive_poly \
  --model-tag=adaptive_poly_2 \
  --ortho-order=2 \
  --ns-steps=1 \
  --muon-norm-iters=0 \
  --print-every=50