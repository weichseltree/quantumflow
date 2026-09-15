#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SWEEP_NAME="transport-beta-sweep"

for beta in 0.0 0.001 0.01 0.1 1.0; do
  run_name="beta_${beta//./_}"
  exp_name="transport-beta-${beta//./_}"
  output_dir="outputs/transport/${run_name}"
  mkdir -p "${output_dir}"

  echo "Launching ${exp_name} (beta=${beta})..."
  exp run "${exp_name}" \
    --prio 2 \
    --sweep "${SWEEP_NAME}" \
    --lane gpu \
    --log "${output_dir}/train.log" \
    -- .venv/bin/python scripts/run_transport.py \
       --beta "${beta}" \
       --steps 1000 \
       --batch-size 256 \
       --output-dir "${output_dir}"
done

echo "All sweep members queued for sweep ${SWEEP_NAME}."
