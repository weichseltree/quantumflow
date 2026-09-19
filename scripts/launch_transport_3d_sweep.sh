#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SWEEP_NAME="transport-3d-beta-sweep"
BASE_DIR="outputs/transport_3d/sweep"
PYTHON="$PWD/.venv/bin/python"
mkdir -p "${BASE_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}; please ensure virtualenv is active." >&2
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "========================================================================"
echo "LAUNCHING 3D OPTIMAL TRANSPORT FLOW MATCHING SWEEP VIA EXPDASH"
echo "Sweep:  ${SWEEP_NAME}"
echo "Output: ${BASE_DIR}"
echo "========================================================================"

queue_run_3d() {
  local exp_name="$1"
  local beta="$2"
  local seed="$3"
  local steps="${4:-1000}"
  local batch_size="${5:-256}"
  local eval_samples="${6:-1024}"

  local out_dir="${BASE_DIR}/${exp_name}"
  mkdir -p "${out_dir}"

  echo "Queueing ${exp_name} (3D OT-CFM, beta=${beta}, seed=${seed})..."
  exp run "${exp_name}" \
    --prio 3 \
    --sweep "${SWEEP_NAME}" \
    --lane gpu \
    --log "${out_dir}/train.log" \
    -- env XLA_PYTHON_CLIENT_PREALLOCATE=false "${PYTHON}" scripts/run_transport_3d.py \
       --beta "${beta}" \
       --seed "${seed}" \
       --steps "${steps}" \
       --batch-size "${batch_size}" \
       --eval-samples "${eval_samples}" \
       --output-dir "${out_dir}"
}

# 3D OT-CFM runs across baseline (beta=0) and isotropic regularized conditions (beta=0.5, 1.0, 2.0)
for seed in 42 43 44; do
  queue_run_3d "transport-3d-b0-s${seed}"   0.0 "${seed}" 800 256 1024
  queue_run_3d "transport-3d-b05-s${seed}"  0.5 "${seed}" 800 256 1024
  queue_run_3d "transport-3d-b1-s${seed}"   1.0 "${seed}" 800 256 1024
  queue_run_3d "transport-3d-b2-s${seed}"   2.0 "${seed}" 800 256 1024
done

echo ""
echo "Queueing final aggregation & summary reporting..."
exp run "transport-3d-finalize" \
  --prio 3 \
  --sweep "${SWEEP_NAME}" \
  --lane cpu \
  --log "${BASE_DIR}/finalize.log" \
  -- "${PYTHON}" scripts/finalize_transport_3d_sweep.py \
     --output-dir "${BASE_DIR}"

echo "All 12 3D transport sweep runs plus finalizer queued under sweep '${SWEEP_NAME}'."

