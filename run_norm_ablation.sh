#!/usr/bin/env bash
set -euo pipefail
export NANOCHAT_BASE_DIR=/workspace/nanochat_cache
mkdir -p logs

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
  echo "COMMON_ARGS: ${COMMON_ARGS[*]}"
  echo "METHOD_ARGS: $*"
  echo "================================================="

  PYTHONUNBUFFERED=1 python -m scripts.base_train \
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
# Matmul-matched-ish to PE T=5: FO-APO T=2
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
# Matmul-matched-ish to PE T=5: SO-APO T=1
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