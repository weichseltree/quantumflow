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

Run the fast validation suite with `python -m pytest`, and lint source files with
`python -m ruff check quantumflow scripts tests`.

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

### Convex kinetic-energy functional

`experiments/convex_1d/hyperparams.yaml` provides an ICNN-based 1D kinetic-energy
functional. Its non-negative hidden and output connections and Softplus activations guarantee
convexity with respect to the discretized density. The functional derivative is obtained with
automatic differentiation; recover an external potential with
`potential_from_kinetic_derivative(derivative, chemical_potential)`, implementing
`v(x) = mu - delta T[n] / delta n(x)`.

## Clean notebooks

To strip cell outputs when notebooks are committed, configure the repository-local filter:

```bash
git config filter.notebook-clean.clean "python clean_notebook.py"
```
