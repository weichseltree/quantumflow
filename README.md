# QuantumFlow — learning the kinetic energy functional of orbital-free DFT

**Archived.** Last worked on **2021-11-21**. TensorFlow 2.x of that era; nothing
here has been re-run since.

Density functional theory is cheap if you know the kinetic energy as a
functional of the density, `T[n]`, and expensive because you do not. This repo
learns it, in the 1D sandbox of
[Snyder et al., *Finding Density Functionals with Machine Learning* (PRL 108,
253002, 2012)](https://doi.org/10.1103/PhysRevLett.108.253002): `N`
non-interacting fermions in a box, under potentials built from a few Gaussian
wells.

What makes it a real test rather than a curve fit is that the *functional
derivative* `δT/δn` is what a DFT calculation actually consumes. A model can fit
`T[n]` well and still be useless, because the derivative it implies is noisy in
the directions the self-consistent density search moves along — which is what
`experiments/self_consistency/` measures.

## The measurement chain

1. `quantumflow/snyder_2012/datasets.py` samples Gaussian-well potentials.
2. `numerov_solver.py` solves the 1D Schrödinger equation for them by Numerov
   shooting, written as a Keras `AbstractRNNCell` so the integration runs
   batched on the GPU and stays differentiable. This is the ground truth.
3. `datasets_dft.py` turns wavefunctions into densities and energies, and can
   subtract the von Weizsäcker functional so the model only has to learn the
   remainder (`subtract_von_weizsaecker` in every experiment YAML).
4. A model predicts `T[n]`; `derivative_model.py` exposes `δT/δn` via autodiff.
5. `self_consistency` runs the density search with the learned derivative.

## The model ladder

| where | model |
|---|---|
| `notebooks/2_kernel_ridge_regresion.ipynb` | kernel ridge regression — the paper's own method, the baseline |
| `quantumflow/snyder_2012/resnet.py` | convolutional ResNet and Fixup-ResNet functionals (`experiments/resnets/`) |
| `quantumflow/xdiff/` | `XdiffPerciever` — Perceiver-style cross-attention keyed on coordinate *differences* rather than absolute positions, so the functional is translation-equivariant by construction (`experiments/xdiff/`) |
| `quantumflow/layers/` | `TrapezoidalIntegral1D`, Fixup multiplier/bias layers |

`quantumflow/ofdft/` is a start on 3D — hydrogen wavefunctions and importance
sampling — that got as far as one notebook and `functions_3d.py`.

## Layout

Datasets and experiments are YAML-instantiated: a node with a `class:` key names
a dotted path, the remaining keys are its kwargs.

- `datasets/<name>/<name>.yaml` — `snyder_2012` (the paper's potentials, read
  from `recreate/paper_potentials.txt`) and `alghadeer_2021` (`3GD`, three
  Gaussians drawn at random). The generated HDF5 is gitignored.
- `experiments/<name>/<name>.yaml` — `resnets`, `xdiff`, `self_consistency`,
  `alghadeer_2021`. Everything under an experiment except the YAML is
  gitignored.
- `notebooks/` — dataset generation, the KRR baseline, per-model training and
  the figure export.

## Running it

Written for Google Colab: put the checkout in Drive under
`Colab Projects/QuantumFlow`, or edit `project_path` in the first cell of each
notebook. Needs `tensorboard>=2.0.0`, `matplotlib`, `ruamel.yaml`, `pandas`.
Some notebooks want a GPU runtime, and each notebook takes its own VM — close
unused sessions under Runtime → Manage Sessions.

## Notebooks are stored without their output

A git clean filter strips outputs on commit. It is per-clone and not committed,
so run this once after cloning:

```
git config filter.clean_notebook.clean $PWD/clean_notebook.py
```

`$PWD` matters. This repo had the path hardcoded to a directory the checkout
left behind years ago, so the filter failed on every `git status` and all
fifteen notebooks showed up as modified by their own stored output — about
4,300 lines of it. `clean_notebook.py` is stdlib-only now, so it does not break
when the interpreter changes.

## Branches

`feature/sin_cos` holds the last six commits — the sin/cos coordinate embedding,
the hydrogen wavefunctions, the `alghadeer_2021` dataset — and `main` has been
fast-forwarded onto it. Both point at the same commit; the branch is kept
because the commit messages refer to it.
