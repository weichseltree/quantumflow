"""Train the convex kinetic-energy functional and score the inversion.

This produces the wing's headline number. The Inversion's claim is that the
Euler equation can be run backwards -- hand the machine a density and it hands
back the potential that shaped it, with no orbitals involved:

    mu = dT_s[n] / dn(x) + v(x)        =>        v(x) = mu - dT_s[n] / dn(x)

The functional is trained on the Snyder benchmark's 100,000 solved systems and
scored on the 1,000 it never saw. The plaque quotes the median, and the
picture is the *worst* of the thousand, not the best.

On the gauge, which decides what the number means. Under the constraint that a
density integrates to its particle number, ``dT/dn`` is fixed only up to an
additive constant: adding a constant to it changes nothing physical, because
the Euler equation absorbs it into mu. So the honest measure of a
reconstruction is the *shape* of the recovered potential, with the best
constant offset removed. The unshifted error is reported beside it, but it
leans on knowing the true chemical potential, which a real inversion would
not, so it is the weaker of the two claims and not the one on the plaque.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax

from quantumflow.expdash import report
from quantumflow.jax.convex import (
    functional_derivative,
    init_icnn,
    kinetic_energy,
    make_composite_training_step,
    save_parameters,
)


def load_split(path: Path, occupied: int) -> dict[str, np.ndarray]:
    """Load a Snyder split and derive what the functional is trained against.

    ``T_s = sum_i eps_i - integral v n`` and ``mu = eps_{N-1}`` are exact for
    the non-interacting system, so both targets come from the solved data
    rather than from any model.
    """
    with h5py.File(path, "r") as handle:
        potential = np.asarray(handle["potential"][()], dtype=np.float64)
        wavefunctions = np.asarray(handle["wavefunctions"][()], dtype=np.float64)
        energies = np.asarray(handle["energies"][()], dtype=np.float64)
        spacing = float(handle.attrs["h"])
        x = np.asarray(handle.attrs["x"], dtype=np.float64)

    density = np.sum(wavefunctions[:, :, :occupied] ** 2, axis=2)
    chemical_potential = energies[:, occupied - 1]
    potential_energy = np.sum(potential * density, axis=1) * spacing
    kinetic = np.sum(energies[:, :occupied], axis=1) - potential_energy
    # The Euler equation, used as a training target rather than as a result.
    derivative = chemical_potential[:, None] - potential

    return {
        "x": x,
        "h": spacing,
        "potential": potential,
        "density": density,
        "kinetic": kinetic,
        "derivative": derivative,
        "chemical_potential": chemical_potential,
    }


def reconstruct(params, data: dict, batch: int = 200) -> dict[str, np.ndarray]:
    """Recover v(x) from each density alone and measure how far off it is."""
    spacing = data["h"]
    densities = jnp.asarray(data["density"], dtype=jnp.float32)

    predicted = []
    for start in range(0, densities.shape[0], batch):
        chunk = densities[start : start + batch]
        predicted.append(np.asarray(functional_derivative(params, chunk, volume_element=spacing)))
    derivative = np.concatenate(predicted, axis=0).astype(np.float64)

    # v = mu - dT/dn, with the true chemical potential. Leans on knowing mu.
    absolute = data["chemical_potential"][:, None] - derivative
    # The same reconstruction with the gauge freedom taken out: what the shape
    # of the well is, which is what the density can actually determine.
    error = absolute - data["potential"]
    shifted = error - np.mean(error, axis=1, keepdims=True)

    depth = data["potential"].max(axis=1) - data["potential"].min(axis=1)
    return {
        "reconstructed": absolute,
        "absolute_pct": 100.0 * np.mean(np.abs(error), axis=1) / depth,
        "shape_pct": 100.0 * np.mean(np.abs(shifted), axis=1) / depth,
        "depth": depth,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/snyder_2012"))
    parser.add_argument("--occupied", type=int, default=1, help="occupied orbitals, N")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--alpha-derivative",
        type=float,
        default=1.0,
        help="weight on the functional-derivative loss, which is what the inversion uses",
    )
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/inversion"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"devices: {jax.devices()}")

    train = load_split(args.dataset_dir / "train" / "dataset.hdf5", args.occupied)
    validate = load_split(args.dataset_dir / "validate" / "dataset.hdf5", args.occupied)
    points = train["density"].shape[1]
    print(f"train {train['density'].shape}  validate {validate['density'].shape}")
    print(f"grid {points} points, h = {train['h']:.6g}")
    print(f"N = {args.occupied}; electrons = {train['density'][0].sum() * train['h']:.4f}")

    density = jnp.asarray(train["density"], dtype=jnp.float32)
    kinetic = jnp.asarray(train["kinetic"], dtype=jnp.float32)
    derivative = jnp.asarray(train["derivative"], dtype=jnp.float32)

    params = init_icnn(jax.random.key(args.seed), input_size=points, hidden_units=args.hidden)
    optimizer = optax.adam(args.lr)
    state = optimizer.init(params)
    step_fn = make_composite_training_step(
        optimizer, alpha_derivative=args.alpha_derivative, volume_element=train["h"]
    )

    key = jax.random.key(args.seed + 1)
    started = time.time()
    history = []
    for step in range(args.steps):
        key, subkey = jax.random.split(key)
        index = jax.random.randint(subkey, (args.batch,), 0, density.shape[0])
        params, state, aux = step_fn(
            params, state, density[index], kinetic[index], derivative[index]
        )

        if step % 50 == 0 or step == args.steps - 1:
            losses = {name: float(value) for name, value in aux.items()}
            history.append({"step": step, **losses})
            report(
                step=step,
                total=args.steps,
                loss=losses["loss"],
                loss_energy=losses["loss_energy"],
                loss_derivative=losses["loss_derivative"],
            )
        if step % 2000 == 0:
            print(
                f"  step {step:>6}  loss {float(aux['loss']):.6g}  "
                f"energy {float(aux['loss_energy']):.6g}  "
                f"derivative {float(aux['loss_derivative']):.6g}"
            )

    elapsed = time.time() - started
    print(f"trained {args.steps} steps in {elapsed / 60:.1f} min")

    scored = reconstruct(params, validate)
    predicted_energy = np.asarray(
        kinetic_energy(params, jnp.asarray(validate["density"], dtype=jnp.float32))
    ).astype(np.float64)
    energy_error = np.abs(predicted_energy - validate["kinetic"])

    shape = scored["shape_pct"]
    absolute = scored["absolute_pct"]
    worst = int(np.argmax(shape))

    summary = {
        "schema": "quantumflow/inversion/1",
        "occupied_orbitals": args.occupied,
        "held_out_systems": int(shape.shape[0]),
        "training": {
            "steps": args.steps,
            "batch": args.batch,
            "learning_rate": args.lr,
            "alpha_derivative": args.alpha_derivative,
            "hidden_units": args.hidden,
            "seed": args.seed,
            "minutes": round(elapsed / 60.0, 2),
            "final_loss": history[-1] if history else {},
        },
        "headline": {
            "statistic": "median absolute error of the reconstructed potential, "
            "as a percentage of each system's own well depth",
            "gauge": "the best constant offset is removed, because dT/dn is fixed "
            "only up to a constant when the density's particle number is held",
            "median_pct": float(np.median(shape)),
            "mean_pct": float(np.mean(shape)),
            "p90_pct": float(np.percentile(shape, 90)),
            "worst_pct": float(np.max(shape)),
        },
        "with_true_chemical_potential": {
            "note": "leans on knowing mu, which a real inversion would not; "
            "reported for completeness, not for the plaque",
            "median_pct": float(np.median(absolute)),
            "worst_pct": float(np.max(absolute)),
        },
        "kinetic_energy": {
            "median_absolute_error_hartree": float(np.median(energy_error)),
            "mean_absolute_error_hartree": float(np.mean(energy_error)),
        },
        "worst_system": {
            "index": worst,
            "shape_pct": float(shape[worst]),
            "well_depth_hartree": float(scored["depth"][worst]),
        },
        "kill_criterion": {
            "threshold_pct": 10.0,
            "passes": bool(np.median(shape) < 10.0),
            "consequence": "above 10% the inversion hangs as a negative result, "
            "with this number on the plaque",
        },
    }

    (args.output_dir / "inversion_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_parameters(params, args.output_dir / "icnn_snyder.npz")
    np.savez_compressed(
        args.output_dir / "worst_case.npz",
        x=validate["x"],
        potential_true=validate["potential"][worst],
        potential_reconstructed=scored["reconstructed"][worst],
        density=validate["density"][worst],
        shape_pct=shape[worst],
    )
    (args.output_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")

    print()
    print(f"MEDIAN SHAPE ERROR: {summary['headline']['median_pct']:.3f}% of well depth")
    head = summary["headline"]
    print(f"  p90 {head['p90_pct']:.3f}%   worst {head['worst_pct']:.3f}%")
    print(f"  with true mu: median {summary['with_true_chemical_potential']['median_pct']:.3f}%")
    mae = summary["kinetic_energy"]["median_absolute_error_hartree"]
    print(f"  kinetic energy MAE: {mae:.6g} Ha")
    print(f"KILL CRITERION (<10%): {'PASSES' if summary['kill_criterion']['passes'] else 'FIRES'}")
    print(f"-> {args.output_dir}")


if __name__ == "__main__":
    main()
