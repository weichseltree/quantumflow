#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pip install --upgrade "jax[cuda13]>=0.4.38"

ORCHARD_ROOT="${ORCHARD_ROOT:-$HOME/weichseltree/orchard}"
if [[ -d "${ORCHARD_ROOT}/packages/tape" ]]; then
  .venv/bin/python -m pip install -e "${ORCHARD_ROOT}/packages/tape"
fi
if [[ -f "${ORCHARD_ROOT}/pyproject.toml" ]]; then
  .venv/bin/python -m pip install -e "${ORCHARD_ROOT}"
fi

.venv/bin/python - <<'PY'
import jax

devices = jax.devices()
print("JAX devices:", devices)
if not any(device.platform == "gpu" for device in devices):
    raise SystemExit("CUDA JAX setup failed: no GPU device is visible")
PY
