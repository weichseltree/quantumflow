#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SWEEP_NAME="convex-3d-piecewise-sweep"
BASE_DIR="outputs/convex_multidim/sweep"
PYTHON="$PWD/.venv/bin/python"
ENV_CMD=(env JAX_PLATFORMS=cpu)
mkdir -p "${BASE_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}; please ensure virtualenv is active." >&2
  exit 1
fi

echo "========================================================================"
echo "LAUNCHING MULTI-DIMENSIONAL CONVEX FUNCTIONAL SWEEP VIA EXPDASH"
echo "Sweep:  ${SWEEP_NAME}"
echo "Output: ${BASE_DIR}"
echo "========================================================================"

# Function to queue an experiment
queue_run() {
  local exp_name="$1"
  local dim="$2"
  local p_type="$3"
  local p_shape="$4"
  local seed="$5"
  local pts="$6"
  local train_sz="$7"
  local test_sz="$8"
  local steps="$9"
  
  local out_dir="${BASE_DIR}/${exp_name}"
  mkdir -p "${out_dir}"
  
  echo "Queueing ${exp_name} (dim=${dim}D, type=${p_type}, shape=${p_shape}, seed=${seed})..."
  exp run "${exp_name}" \
    --prio 2 \
    --sweep "${SWEEP_NAME}" \
    --lane cpu \
    --log "${out_dir}/train.log" \
    -- "${ENV_CMD[@]}" "${PYTHON}" scripts/run_convex_multidim.py \
       --dimension "${dim}" \
       --potential-type "${p_type}" \
       --potential-shape "${p_shape}" \
       --grid-points "${pts}" \
       --orbitals 2 \
       --train-size "${train_sz}" \
       --test-size "${test_sz}" \
       --steps "${steps}" \
       --batch-size 16 \
       --seed "${seed}" \
       --output-dir "${out_dir}"
}

# 1. 3D Piecewise Constant Potentials (grid 10^3 = 1000 pts)
queue_run "3d-pw-box-s0"       3 "piecewise_constant" "box"       42 10 50 15 250
queue_run "3d-pw-box-s1"       3 "piecewise_constant" "box"       43 10 50 15 250
queue_run "3d-pw-staircase-s0" 3 "piecewise_constant" "staircase" 42 10 50 15 250
queue_run "3d-pw-sphere-s0"    3 "piecewise_constant" "sphere"    42 10 50 15 250

# 2. 3D Piecewise Linear Potentials (grid 10^3 = 1000 pts)
queue_run "3d-pl-polyhedral-s0" 3 "piecewise_linear" "convex_polyhedral" 42 10 50 15 250
queue_run "3d-pl-polyhedral-s1" 3 "piecewise_linear" "convex_polyhedral" 43 10 50 15 250
queue_run "3d-pl-pyramid-s0"    3 "piecewise_linear" "l1_pyramid"        42 10 50 15 250

# 3. 3D Gaussian Baseline (grid 10^3 = 1000 pts)
queue_run "3d-gaussian-s0"     3 "gaussian"           "box"       42 10 50 15 250
queue_run "3d-gaussian-s1"     3 "gaussian"           "box"       43 10 50 15 250

# 4. 2D Comparisons (grid 24^2 = 576 pts)
queue_run "2d-pw-box-s0"       2 "piecewise_constant" "box"               42 24 60 20 300
queue_run "2d-pl-polyhedral-s0" 2 "piecewise_linear" "convex_polyhedral" 42 24 60 20 300
queue_run "2d-gaussian-s0"     2 "gaussian"           "box"               42 24 60 20 300

echo ""
echo "Queueing final aggregation & summary reporting..."
exp run "sweep-finalize" \
  --prio 2 \
  --sweep "${SWEEP_NAME}" \
  --lane cpu \
  --log "${BASE_DIR}/finalize.log" \
  -- "${ENV_CMD[@]}" "${PYTHON}" scripts/finalize_convex_multidim_sweep.py \
     --output-dir "${BASE_DIR}"

echo "All 12 sweep experiments plus finalizer queued under sweep '${SWEEP_NAME}'."
