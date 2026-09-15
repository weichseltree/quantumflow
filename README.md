# QuantumFlow

Research tooling for QuantumFlow and orbital-free density-functional-theory experiments.

## Layout

| Path | Purpose |
| --- | --- |
| `quantumflow/` | Reusable Python package and experiment CLI. |
| `experiments/` | Versioned YAML configurations and small input fixtures. |
| `notebooks/` | Exploratory analysis notebooks. |
| `scripts/` | Backward-compatible wrappers for the installed CLI. |
| `outputs/` | Generated datasets, checkpoints, and model exports (ignored by Git). |

## Setup

Python 3.10 or newer is required. Create an environment and install the package with its
developer tools:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The active modeling stack is JAX and Optax. Legacy TensorFlow/Keras modules remain available
only during migration; install `python -m pip install -e ".[legacy-tensorflow]"` when running
an unreworked legacy experiment.

Run the complete local validation suite before opening a pull request:

```bash
python -m pip check
python -m pytest
python -m ruff check quantumflow/cli.py quantumflow/noninteracting_1d/convex.py tests
python -m ruff format --check quantumflow/cli.py quantumflow/noninteracting_1d/convex.py tests
```

## Running experiments

Configurations live under `experiments/<experiment>/` and may use either
`hyperparams.yaml` or a single descriptive YAML file. Run artifacts are written under
`outputs/<experiment>/<run-name>/` by default:

```bash
quantumflow-dataset snyder_2012 recreate_dataset
quantumflow-train resnets resnet_100 --output-dir /path/to/results
```

The legacy `scripts/generate_dataset.py` and `scripts/train_network.py` commands remain
available and accept the same arguments.

### Experiment dashboards

Each invocation has a stable artifact location at
`outputs/<experiment>/<run-name>/`. Use a unique run name for each trial and pass a common
`--output-dir` when a dashboard needs to aggregate multiple runs. TensorBoard callbacks in
an experiment configuration write event files into that same run directory.

ExpDash runs in WSL at `http://localhost:8686/`. Launch local experiments from the WSL
checkout (`~/weichseltree/quantumflow`) through its lock-aware launcher, not from Windows:

```bash
cd ~/weichseltree/quantumflow
mkdir -p outputs/transport/example
EXP_NAME=transport-example EXP_PRIO=2 \
EXP_LOG="$PWD/outputs/transport/example/train.log" \
exp run transport-example --lane gpu -- .venv/bin/python path/to/runner.py
```

`exp run` serializes GPU work, creates the ExpDash status record, and exports
`EXP_METRICS_FILE`. Training code can then publish run progress with the built-in bridge:

```python
from quantumflow.expdash import report

report(step=step, total=total_steps, loss=float(loss), penalty=float(penalty))
```

The bridge atomically writes the metrics JSON expected by ExpDash, including history for
curves and ETA calculation. It returns `False` outside an `exp run` process, allowing the
same runner to work without the dashboard. Keep configurations, source code, and dashboard
metadata in Git; keep checkpoints, event logs, and generated datasets in `outputs/`, which
is intentionally ignored.

For local inspection without a third-party dashboard:

```bash
tensorboard --logdir outputs
```

### Convex kinetic-energy functional

The active convex-functional implementation is JAX-first:

```python
import jax
from quantumflow.jax import functional_derivative, init_icnn, kinetic_energy

params = init_icnn(jax.random.key(0), input_size=grid_points)
energy = kinetic_energy(params, density)
derivative = functional_derivative(params, density)
```

Its ICNN parameterization applies non-negative hidden/output connections and Softplus
activations, guaranteeing convexity with respect to the discretized density. Use
`potential_from_kinetic_derivative(derivative, chemical_potential)` to implement
`v(x) = mu - delta T[n] / delta n(x)`. Use `make_training_step(optimizer)` to obtain a
JIT-compiled update function for a fixed Optax optimizer.

## Clean notebooks

To strip cell outputs when notebooks are committed, configure the repository-local filter:

```bash
git config filter.notebook-clean.clean "python clean_notebook.py"
```
