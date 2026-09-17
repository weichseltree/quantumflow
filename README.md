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
python -m pip install -e orchard/packages/score -e orchard/packages/tape -e orchard
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
exp run transport-example \
  --prio 2 \
  --sweep transport-example \
  --lane gpu \
  --log "$PWD/outputs/transport/example/train.log" \
  -- .venv/bin/python path/to/runner.py
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

### The Orchard exhibition

The quantumflow wing of the Mind Palace is built from this repository. Its
manifest is `orchard.yaml`, which is canonical: the fund's `trees/quantumflow.yaml`
is a mirror regenerated from it. Install the extra first:

```bash
python -m pip install -e ".[exhibition]"
```

Everything the exhibition needs is computed on the CPU from the exact solver;
only the inversion thesis needs the GPU lane. Pin JAX to the CPU so an export
never opens a CUDA context outside `gpurun`, where it would contend invisibly
with whatever holds the lane:

```bash
# 25 glTF models: signed orbital lobes, density shells, the potential relief,
# the convexity pair, and the bond-separation walk.
exp run quantumflow-models --prio 10 --lane cpu -- \
  .venv/bin/python scripts/export_orchard_models.py --which all --output-dir results/models

# The Shooting Gallery tape. Its scrub axis is trial energy, not time.
exp run quantumflow-shooting-tape --prio 10 --lane cpu -- \
  .venv/bin/python scripts/export_shooting_tape.py --output-dir results/tape/shooting

# One equation still per room, plus the wall lines for translation.
exp run quantumflow-stills --prio 10 --lane cpu -- \
  .venv/bin/python scripts/render_equation_stills.py --output-dir results/stills
```

Artefacts land in `results/`, which is ignored: Orchard bundles them from the
working tree and records their content hashes, so the bundle is the durable
record. Each exporter writes a manifest naming what the room's plaque must
state — the metres-per-Hartree of the orbital ladder, the Hartree-per-unit of
the tape's compressed energy axis, the containment fraction each density shell
actually encloses. Those constants are not decoration: unstated, a height on a
wall means nothing.

Two caveats travel with the artefacts and belong on the plaques. The flow
tape's third coordinate is the learned potential up to an arbitrary additive
gauge, so tapes from independently trained models cannot be compared by
height. And the shooting tape is not a movie — each frame is an independent
trial energy, in increasing order.

`quantumflow.glb` writes binary glTF directly, in core glTF 2.0 with no
extensions, so the grove never needs a decoder it does not host. Vertex colour
is the load-bearing feature: every surface carries a measured quantity rather
than a palette.

## Clean notebooks

To strip cell outputs when notebooks are committed, configure the repository-local filter:

```bash
git config filter.notebook-clean.clean "python clean_notebook.py"
```
