#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SWEEP_NAME="transport-beta-pilot"
BASE_DIR="outputs/transport/pilot"
PYTHON="$PWD/.venv/bin/python"
JAX_ENV=(env XLA_PYTHON_CLIENT_PREALLOCATE=false)
mkdir -p "${BASE_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}; run scripts/setup_pilot_wsl.sh first." >&2
  exit 1
fi

echo "========================================================================"
echo "LAUNCHING ISOTROPIC-HESSIAN OT-CFM MULTI-SEED BETA PILOT VIA EXPDASH"
echo "Sweep: ${SWEEP_NAME}"
echo "Output: ${BASE_DIR}"
echo "========================================================================"

# Matrix:
# Betas: 0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1
# Seeds: 0, 1, 2
# BS: 256
# Ablation Betas: 0, 1e-3, 1e-2 for BS: 128

declare -a BETAS=("0" "0.0001" "0.0003" "0.001" "0.003" "0.01" "0.03" "0.1" "0.3" "1.0")
declare -a SEEDS=("0" "1" "2")

# Main grid: BS 256
for beta in "${BETAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_name="pilot_beta_${beta//./_}_s${seed}_bs256"
    exp_name="pilot-b${beta//./_}-s${seed}-bs256"
    output_dir="${BASE_DIR}/${run_name}"
    mkdir -p "${output_dir}"

    echo "Queueing ${exp_name} (beta=${beta}, seed=${seed}, bs=256)..."
    exp run "${exp_name}" \
      --prio 2 \
      --sweep "${SWEEP_NAME}" \
      --lane gpu \
      --log "${output_dir}/train.log" \
      -- "${JAX_ENV[@]}" "${PYTHON}" scripts/run_transport.py \
         --beta "${beta}" \
         --seed "${seed}" \
         --batch-size 256 \
         --steps 2000 \
         --output-dir "${output_dir}"
  done
done

# Ablation grid: BS 128
declare -a ABLATION_BETAS=("0" "0.001" "0.01")
for beta in "${ABLATION_BETAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_name="pilot_beta_${beta//./_}_s${seed}_bs128"
    exp_name="pilot-b${beta//./_}-s${seed}-bs128"
    output_dir="${BASE_DIR}/${run_name}"
    mkdir -p "${output_dir}"

    echo "Queueing ${exp_name} (beta=${beta}, seed=${seed}, bs=128)..."
    exp run "${exp_name}" \
      --prio 2 \
      --sweep "${SWEEP_NAME}" \
      --lane gpu \
      --log "${output_dir}/train.log" \
      -- "${JAX_ENV[@]}" "${PYTHON}" scripts/run_transport.py \
         --beta "${beta}" \
         --seed "${seed}" \
         --batch-size 128 \
         --steps 2000 \
         --output-dir "${output_dir}"
  done
done

echo ""
echo "Queueing final aggregation, report, and Orchard gallery export..."
exp run "pilot-finalize" \
  --prio 2 \
  --sweep "${SWEEP_NAME}" \
  --lane gpu \
  --log "${BASE_DIR}/finalize.log" \
  -- "${JAX_ENV[@]}" "${PYTHON}" scripts/finalize_beta_pilot.py

echo "All 39 pilot runs plus finalizer queued under sweep '${SWEEP_NAME}'."
