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
python -m pytest -m "not orchard"
python -m ruff check quantumflow/cli.py quantumflow/noninteracting_1d/convex.py tests
python -m ruff format --check quantumflow/cli.py quantumflow/noninteracting_1d/convex.py tests
```

Orchard export is an optional integration supported on Python 3.12 or newer.
The official Orchard and `orchard-tape` packages are maintained in the
[`weichseltree/orchard`](https://github.com/weichseltree/orchard) repository,
not published as QuantumFlow dependencies. To run the integration tests, clone
that repository alongside this checkout and install its workspace packages:

```bash
git clone https://github.com/weichseltree/orchard.git
python -m pip install -e orchard/packages/tape -e orchard
python -m pytest -m orchard tests/test_orchard_export.py
```

The regular test command deliberately excludes these marked tests so supported
Python 3.10 and 3.11 environments do not require the Orchard stack.

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

### Isotropic-Hessian OT-CFM beta pilot

The multi-seed pilot in issue #11 uses a matched evaluation stream across all
conditions, exact and sliced Wasserstein metrics, mode-coverage diagnostics,
Hessian eigenvalue-spread metrics, and an RK4 step-count Pareto analysis.
Prepare the WSL environment and queue all 39 runs plus the final report job:

```bash
./scripts/setup_pilot_wsl.sh
./scripts/launch_expdash_pilot.sh
exp board
```

Every run is uniquely named and grouped under the
`transport-beta-pilot` ExpDash sweep. The final queued job writes
`outputs/transport/pilot/pilot_analysis.json`, `PILOT_REPORT.md`, a portable
model archive for every run, and `gallery/` containing a WebXR viewer,
`video/tape/1` particle tape, and `orchard/bundle/1` bundle for the promoted
candidate.

ExpDash progress uses cumulative samples processed as its x-axis. Continuation
runs also publish their absolute optimizer step and accept independent
`--step-offset` and `--sample-offset` values, so charts remain monotonic even
when a promoted run changes batch size.

An optional CPU-lane controller dynamically raises the priority of informative
middle-range beta values, then performs successive halving without changing
the fixed pilot: the baseline and two leaders continue to 4,000 steps, and
the baseline plus leader continue to 8,000 steps. Each continuation starts
from the previous portable model archive and logs cumulative samples.

```bash
exp run pilot-adaptive-controller --prio 5 \
  --sweep transport-beta-adaptive --lane cpu \
  --log outputs/transport/adaptive/controller.log \
  -- .venv/bin/python scripts/adaptive_beta_controller.py
```

The launcher disables JAX's default whole-device memory preallocation. This
keeps the experiments within the 8 GiB RTX 3070 budget while ExpDash retains
exclusive scheduling of the GPU lane.

### Quick gallery demo (no full pilot required)

To exercise the export pipeline and preview sample 2D/3D/WebXR/Orchard
artifacts without running the 39-run beta pilot, train a tiny model and
export a demo gallery in one command:

```bash
python scripts/generate_demo_gallery.py --output-dir outputs/demo_gallery/demo
```

This trains a small OT-CFM model (a few seconds on CPU, ~300 steps by
default) to `outputs/demo_gallery/demo/model.npz`, then writes:

- `gallery/tape/` -- a `video/tape/1` trajectory tape plus `webxr_particles.json`
- `gallery/index.html` -- an offline-capable WebXR viewer (vendored Three.js assets)
- `gallery/bundles/` -- a verified `orchard/bundle/1` bundle (skip with `--no-bundle`)
- `gallery/gallery.json` -- a manifest recording provenance and file locations,
  in the same `quantumflow/gallery/2` schema used by the full pilot's
  representative gallery

Building the tape, viewer, and bundle requires the optional Orchard stack
described above; if it isn't installed the command fails with an explicit
message naming the missing package. Pass `--skip-train` to reuse an existing
`model.npz` instead of retraining, and `--no-bundle` to skip the bundle step.

Serve the gallery locally to view it in a browser (or a WebXR headset on the
same network):

```bash
python -m http.server --directory outputs/demo_gallery/demo/gallery 8000
```

Then open `http://localhost:8000/` (or the host machine's LAN address from a
headset). `outputs/` is intentionally gitignored, so demo artifacts are never
committed. To publish a gallery -- for example via GitHub Pages -- copy the
contents of `gallery/` into the target branch or a workflow's static-hosting
artifact directory rather than committing it under `outputs/`.

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
`v(x) = mu - delta T[n] / delta n(x)`. Use `make_training_step(optimizer)` or
`make_composite_training_step(optimizer)` to obtain a JIT-compiled update function for joint
energy and functional derivative training.

Multi-dimensional (>1D) experiments in 2D and 3D can be run directly via:

```bash
python scripts/run_convex_multidim.py --dimension 2 --grid-points 24 --orbitals 3
python scripts/run_convex_multidim.py --dimension 3 --grid-points 10 --orbitals 2
```

### Visualizations, Videos & Grove 3D Exhibition

QuantumFlow includes visualization tools and export pipelines for figures, video animations, and
spatial exhibitions:

```bash
# Render publication-ready figures (PNG, SVG)
python scripts/render_figures.py --output-dir outputs/figures

# Render animated videos (GIF, MP4) of variational density relaxation & OT transport
python scripts/render_video.py --output-dir outputs/videos --type all --format gif

# Build complete 3D Grove exhibition package (figures, videos, and glowing WebGL volumetric shaders)
python scripts/export_grove_exhibition.py --output-dir outputs/grove_exhibition
```

## Clean notebooks

To strip cell outputs when notebooks are committed, configure the repository-local filter:

```bash
git config filter.notebook-clean.clean "python clean_notebook.py"
```
