"""Export the quantumflow exhibition's glTF models for the grove.

Writes one ``.glb`` per exhibit, plus a manifest carrying what each plaque has
to say and where the model should stand. The grove hangs each glb as its own
``model`` exhibit, so the per-exhibit budget (8 MB, 150,000 triangles, 256
draw calls) applies to each file separately and the row down the orangery is
placement, not geometry.

Everything here runs on the CPU. The lane lock only serialises what goes
through ``gpurun``, so a JAX import that grabs a CUDA context would contend
invisibly with whatever training holds the lane; the pin below is what keeps
this an ordinary CPU job.
"""

from __future__ import annotations

import os

# Before JAX is imported anywhere down the import graph.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# The convexity chord test compares numbers of order 1e4 against numbers of
# order 1; single precision turns that comparison into noise.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse  # noqa: E402
import json  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from quantumflow.exhibition import (  # noqa: E402
    convexity_bowl_models,
    density_shell_model,
    orbital_model,
    relief_model,
    solve_showcase_system,
)
from quantumflow.glb import Model, write_glb  # noqa: E402
from quantumflow.multidim import Grid, solve_multidim_schroedinger  # noqa: E402

#: The bond separations the visitor walks past, in bohr. Seven stations over
#: the orangery's 42 m is a station every six metres: far enough apart that
#: two are never read as one object, close enough that the splitting reads as
#: continuous as you walk.
WALK_SEPARATIONS = (0.4, 0.9, 1.4, 2.0, 2.6, 3.4, 4.4)

#: The band of heights the energy-ladder placement may use, in metres above
#: the orangery floor. Chosen to sit between waist and well below the 10.5 m
#: glass, so the whole ladder is in one view from the middle of the room.
LADDER_LOW = 2.0
LADDER_HIGH = 7.0


def _emit(model: Model, report, output_dir: Path, entries: list[dict], **extra) -> None:
    path = write_glb(output_dir / f"{model.name}.glb", model)
    entry = {
        "file": path.name,
        "bytes": path.stat().st_size,
        **asdict(report),
        **extra,
    }
    entries.append(entry)
    print(
        f"  {path.name:<34} {report.triangles:>7,} tris  "
        f"{len(model.surfaces):>2} draws  {entry['bytes'] / 1e6:.2f} MB"
    )


def build_walk(output_dir: Path, points: int, refine: int) -> list[dict]:
    """The bond-separation row: walking the orangery is pulling a molecule apart."""
    print(f"walk: {len(WALK_SEPARATIONS)} separations at {points} points per axis")
    entries: list[dict] = []
    stations = []

    for separation in WALK_SEPARATIONS:
        solution = solve_showcase_system(points=points, separation=separation, num_orbitals=2)
        stations.append((separation, solution))
        print(
            f"  d = {separation:.1f} bohr  "
            f"E0 = {solution['orbital_energies'][0]:+.4f}  "
            f"E1 = {solution['orbital_energies'][1]:+.4f}  "
            f"split = {solution['orbital_energies'][1] - solution['orbital_energies'][0]:.4f} Ha"
        )

    energies = np.array([e for _, s in stations for e in s["orbital_energies"][:2]])
    span = float(energies.max() - energies.min())
    # One constant for the whole row, so heights are comparable between
    # stations. The plaque states it; without it the ladder is decoration.
    metres_per_hartree = (LADDER_HIGH - LADDER_LOW) / span if span > 0 else 0.0

    for separation, solution in stations:
        for index in (0, 1):
            energy = float(solution["orbital_energies"][index])
            model, report = orbital_model(
                solution["grid"],
                solution["wavefunctions"][:, index],
                energy_hartree=energy,
                index=index,
                refine=refine,
            )
            model.name = f"walk-d{separation:.1f}-orbital{index}".replace(".", "p")
            height = LADDER_LOW + (energy - float(energies.min())) * metres_per_hartree
            _emit(
                model,
                report,
                output_dir,
                entries,
                station={
                    "separation_bohr": separation,
                    "state": "bonding" if index == 0 else "antibonding",
                    "energy_hartree": energy,
                    "height_m": round(height, 3),
                    "metres_per_hartree": round(metres_per_hartree, 4),
                },
            )
    return entries


def build_standing(output_dir: Path, points: int, refine: int) -> list[dict]:
    """The four standing objects: orbitals, the density, the relief."""
    entries: list[dict] = []

    print(f"orbitals + density: one solve at {points} points per axis")
    # `close_shell` because the density shells are summed over the occupied
    # set: ending the sum inside a degenerate multiplet would make the cloud's
    # shape depend on a rotation the solver picked, and that cloud is the
    # centrepiece of The Cloud.
    solution = solve_showcase_system(
        points=points, separation=1.6, num_orbitals=6, close_shell=True
    )
    energies = solution["orbital_energies"]
    print(f"  occupied {solution['occupied']} of 6 requested (closed shell)")
    print("  energies (Ha):", np.round(energies, 4))
    if solution["degenerate_cut"]:
        raise SystemExit("refusing to export: the occupied set splits a degenerate multiplet")
    span = float(energies.max() - energies.min())
    metres_per_hartree = (LADDER_HIGH - LADDER_LOW) / span if span > 0 else 0.0

    for index in range(len(energies)):
        model, report = orbital_model(
            solution["grid"],
            solution["wavefunctions"][:, index],
            energy_hartree=float(energies[index]),
            index=index,
            refine=refine,
        )
        height = LADDER_LOW + (float(energies[index]) - float(energies.min())) * metres_per_hartree
        _emit(
            model,
            report,
            output_dir,
            entries,
            station={
                "energy_hartree": float(energies[index]),
                "height_m": round(height, 3),
                "metres_per_hartree": round(metres_per_hartree, 4),
                "nodes_visible": "count the gaps between lobes",
            },
        )

    model, report = density_shell_model(solution["grid"], solution["density"], refine=refine)
    _emit(model, report, output_dir, entries)

    print("potential relief: 2D solve")
    grid_2d = Grid.create(dimension=2, lower=-3.2, upper=3.2, points=96)
    coords = grid_2d.coordinates
    offset = np.array([0.8, 0.0])
    potential = -12.0 * (
        np.exp(-np.sum((coords + offset) ** 2, axis=-1) / (2 * 1.2**2))
        + np.exp(-np.sum((coords - offset) ** 2, axis=-1) / (2 * 1.2**2))
    )
    solution_2d = solve_multidim_schroedinger(potential, grid_2d, num_orbitals=4)

    model, report = relief_model(
        grid_2d, potential, name="potential-relief", relief_m=0.5, width_m=3.2, invert=True
    )
    _emit(model, report, output_dir, entries)

    model, report = relief_model(
        grid_2d,
        solution_2d["density"],
        name="density-relief",
        relief_m=0.35,
        width_m=3.2,
    )
    _emit(model, report, output_dir, entries)
    return entries


def build_bowls(output_dir: Path, dataset: Path, samples: int) -> list[dict]:
    """The convexity pair, from real benchmark densities."""
    print(f"bowls: affine slice through {dataset}")
    with h5py.File(dataset, "r") as handle:
        wavefunctions = handle["wavefunctions"][:3]
        spacing = float(handle.attrs["h"])
    densities = np.sum(wavefunctions**2, axis=2)
    base = densities[0]
    print(f"  base density integrates to {base.sum() * spacing:.4f} electrons")

    entries: list[dict] = []
    for model, report in convexity_bowl_models(
        base, densities[1] - densities[0], densities[2] - densities[0], samples=samples
    ):
        _emit(model, report, output_dir, entries)
        print(f"    violating chord samples: {report.plaque['violating_samples']}")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--which",
        default="standing",
        choices=("standing", "walk", "bowls", "all"),
        help="which family of models to export",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/models"))
    parser.add_argument(
        "--points", type=int, default=48, help="grid points per axis for the 3D solves"
    )
    parser.add_argument(
        "--refine", type=int, default=2, help="display-only refinement of the isosurface"
    )
    parser.add_argument("--samples", type=int, default=96, help="bowl slice resolution")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/snyder_2012/validate/dataset.hdf5"),
        help="benchmark densities the convexity slice is taken through",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    if args.which in ("standing", "all"):
        entries += build_standing(args.output_dir, args.points, args.refine)
    if args.which in ("walk", "all"):
        entries += build_walk(args.output_dir, args.points, args.refine)
    if args.which in ("bowls", "all"):
        entries += build_bowls(args.output_dir, args.dataset, args.samples)

    manifest = args.output_dir / f"manifest-{args.which}.json"
    manifest.write_text(json.dumps({"models": entries}, indent=2), encoding="utf-8")

    total = sum(entry["bytes"] for entry in entries)
    print(f"\n{len(entries)} models, {total / 1e6:.2f} MB total -> {args.output_dir}")
    over = [e["file"] for e in entries if e["bytes"] > 8_000_000 or e["triangles"] > 150_000]
    if over:
        print(f"OVER BUDGET: {', '.join(over)}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
