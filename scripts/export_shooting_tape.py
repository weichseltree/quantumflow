"""Write the Shooting Gallery's tape: a scrub bar that is an energy dial.

Every other tape in the grove plays time. This one does not, and that is the
whole exhibit. Each frame is one trial energy; scrubbing sweeps the energy
upward through the well. At almost every setting the wave fails to reach the
far wall and collapses into a flat line with a runaway spike at the end. At a
handful of settings it lands, and a standing wave fills the box whose nodes a
visitor can count.

Nobody is told that energy is quantized. They find the settings that work.

The tape's meta, its units string and the room's wall line all say that the
axis is trial energy rather than time, because a visitor who reads it as time
learns the opposite of the lesson.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import orchard_tape  # noqa: E402

from quantumflow.shooting import sweep  # noqa: E402

#: The three things in the room, as tape species.
WELL, DIAL, WAVE = 0, 1, 2
SPECIES_NAMES = ["The well", "The trial energy", "The wave"]

#: Tape units per Hartree on the vertical axis: one unit stands for 10 Hartree.
#:
#: The energy axis is a display convention exactly as the wave's amplitude is,
#: and it has to be compressed or the exhibit fights the room. At true 1:1 the
#: box came out 2.4x taller than long, and a tape is scaled so its LONGEST side
#: measures the hung length -- so running the wave 34 m down the Shooting
#: Gallery would have put the energy axis 80 m up through a 10.5 m ceiling, and
#: fitting the height instead would have squeezed the whole plot into 4.2 m and
#: wasted the 42 m of run the room was sized for. At this scale the box is
#: 21.2 x 5.5 x 2.1 and hangs as 34 m long, 8.8 m tall.
#:
#: Being a convention does not make it free: the constant goes in the meta and
#: on the plaque, like the metres-per-Hartree of the orbital ladder. Unstated,
#: a height is decoration.
ENERGY_UNITS_PER_HARTREE = 0.10

#: How far the wave's crest reaches, in tape units, riding on its trial-energy
#: line. Held at about a third of this well's depth so the wiggle stays legible
#: without swamping the landscape it sits in. An eigenfunction's amplitude is
#: arbitrary -- the normalisation is a convention, not a measurement -- so this
#: is a display choice on a different footing from the energy scale above, and
#: the two are kept as separate numbers for that reason.
WAVE_AMPLITUDE_UNITS = 0.5
#: Strands across the ribbon, so the wave reads as an object rather than a line.
RIBBON_STRANDS = 5
RIBBON_DEPTH = 2.0
#: The well runs 0 to 1 in the benchmark; the room is long.
LENGTH_UNITS = 20.0


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/snyder_2012/validate/dataset.hdf5"),
        help="the benchmark the well is taken from",
    )
    parser.add_argument("--system", type=int, default=0, help="which held-out system")
    parser.add_argument("--trials", type=int, default=420, help="frames, one per trial energy")
    parser.add_argument("--levels", type=int, default=2, help="how many eigenvalues to sweep past")
    parser.add_argument("--output-dir", type=Path, default=Path("results/tape/shooting"))
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as handle:
        potential = np.asarray(handle["potential"][args.system], dtype=np.float64)
        reference = np.asarray(handle["energies"][args.system], dtype=np.float64)
        x = np.asarray(handle.attrs["x"], dtype=np.float64)
        # The generator's own spacing, not x[1] - x[0]; see shooting.sweep.
        spacing = float(handle.attrs["h"])

    levels = min(args.levels, reference.shape[0])
    energy_min = float(potential.min()) - 1.0
    # Stop a little past the last eigenvalue the sweep is meant to find, so the
    # visitor sees the wave fail again on the far side of it.
    energy_max = float(reference[levels - 1]) + 0.35 * float(
        reference[levels - 1] - reference[0]
    )
    print(f"system {args.system}: well from {potential.min():.3f} to {potential.max():.3f} Ha")
    print(f"sweeping {args.trials} trial energies over [{energy_min:.3f}, {energy_max:.3f}] Ha")

    result = sweep(
        potential,
        x,
        energy_min=energy_min,
        energy_max=energy_max,
        trials=args.trials,
        h=spacing,
    )
    print(f"eigenvalues found by shooting: {np.round(result.eigen_energies, 6)}")
    print(f"benchmark's stored values:     {np.round(reference[:levels], 6)}")
    if result.eigen_energies.size >= levels:
        error = np.abs(result.eigen_energies[:levels] - reference[:levels]).max()
        print(f"max discrepancy: {error:.3e} Ha")

    points = x.shape[0]
    x_units = x * LENGTH_UNITS
    dial_x = np.linspace(0.0, LENGTH_UNITS, 80)

    well_count = points
    dial_count = dial_x.shape[0]
    wave_count = points * RIBBON_STRANDS
    total = well_count + dial_count + wave_count

    species = np.concatenate(
        [
            np.full(well_count, WELL, dtype=np.uint8),
            np.full(dial_count, DIAL, dtype=np.uint8),
            np.full(wave_count, WAVE, dtype=np.uint8),
        ]
    )
    strand_z = np.repeat(np.linspace(-RIBBON_DEPTH / 2, RIBBON_DEPTH / 2, RIBBON_STRANDS), points)
    wave_x = np.tile(x_units, RIBBON_STRANDS)

    well_y = potential * ENERGY_UNITS_PER_HARTREE

    frames: list[np.ndarray] = []
    for index in range(args.trials):
        energy = float(result.energies[index])
        dial_y = energy * ENERGY_UNITS_PER_HARTREE
        wave_y = dial_y + WAVE_AMPLITUDE_UNITS * np.tile(result.waves[index], RIBBON_STRANDS)
        frame = np.concatenate(
            [
                np.column_stack([x_units, well_y, np.zeros(points)]),
                np.column_stack([dial_x, np.full(dial_count, dial_y), np.zeros(dial_count)]),
                np.column_stack([wave_x, wave_y, strand_z]),
            ]
        )
        frames.append(frame)

    stacked = np.concatenate(frames, axis=0)
    lower = stacked.min(axis=0)
    upper = stacked.max(axis=0)
    span = upper - lower
    padding = np.maximum(span * 0.03, 1e-3)
    extent = span + 2.0 * padding
    origin = lower - padding

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "title": "The Shooting Gallery: quantization found by hand",
        "description": (
            "One frame per trial energy. The wave is integrated outward from the left "
            "wall by the Numerov recurrence; at almost every energy it fails to reach "
            "the right wall."
        ),
        "species_names": SPECIES_NAMES,
        "axis_warning": (
            "THE SCRUB AXIS IS TRIAL ENERGY, NOT TIME. Nothing in this tape is moving. "
            "Each frame is an independent guess at the energy, in increasing order."
        ),
        "coordinate_convention": {
            "scientific": ["position in the well", "energy (Hartree)", "ribbon depth"],
            "tape": "scientific - tape_origin, vertical axis scaled by energy_units_per_hartree",
            "tape_origin": origin.tolist(),
            "viewer": ["x", "energy", "depth"],
        },
        "display": {
            "energy_units_per_hartree": ENERGY_UNITS_PER_HARTREE,
            "hartree_per_tape_unit": 1.0 / ENERGY_UNITS_PER_HARTREE,
            "energy_axis_is_compressed": (
                f"one tape unit on the vertical axis stands for "
                f"{1.0 / ENERGY_UNITS_PER_HARTREE:.1f} Hartree; the plaque must say so, and "
                "the exhibit's Hartree-per-metre follows from the hung length"
            ),
            "wave_amplitude_units": WAVE_AMPLITUDE_UNITS,
            "wave_normalisation": (
                "each frame's wave is divided by its own largest magnitude, so a diverging "
                "trial reads as a flat line with a spike at the far wall"
            ),
            "amplitude_is_a_convention": (
                "an eigenfunction amplitude carries no physics; the height of the wiggle "
                "is a display choice, the height of the line it rides on is the energy"
            ),
            "length_units_per_box": LENGTH_UNITS,
        },
        "physics": {
            "equation": "psi'' = 2 (v - E) psi, integrated by Numerov, psi(0) = 0",
            "eigenvalues_found_hartree": result.eigen_energies.tolist(),
            "benchmark_energies_hartree": reference[:levels].tolist(),
            "benchmark": "Snyder et al. 2012 one-dimensional benchmark, held-out split",
        },
        "provenance": {
            "dataset": str(args.dataset),
            "system_index": args.system,
            "synthetic": False,
        },
    }

    with orchard_tape.TapeWriter(
        path=args.output_dir,
        box=tuple(float(value) for value in extent),
        n_total=total,
        run_seed=args.system,
        quantize="uint16",
        periodic=(False, False, False),
        velocity=False,
        scalars=(("species", "uint8"), ("endpoint", "float32")),
        units="position in a unit box; energy in Hartree; frames indexed by trial energy",
        t0_origin="lowest trial energy in the sweep",
        t0_offset_tau=0.0,
        meta=meta,
        git_sha=git_sha(),
    ) as writer:
        for index in range(args.trials):
            writer.append(
                step=index,
                time=float(result.energies[index]),
                pos=frames[index] - origin,
                species=species,
                # How far this trial's wave finishes from the far wall: zero
                # exactly at an eigenvalue. The quantity the room is about.
                endpoint=np.full(total, result.endpoint[index], dtype=np.float32),
            )

    summary = {
        "frames": args.trials,
        "particles": total,
        "energy_range_hartree": [energy_min, energy_max],
        "eigenvalues_found": result.eigen_energies.tolist(),
        "benchmark_energies": reference[:levels].tolist(),
    }
    (args.output_dir / "shooting_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    aspect = extent[1] / extent[0]
    print(f"\nbox {[round(float(v), 2) for v in extent]}  (height/length = {aspect:.3f})")
    print(f"vertical axis: 1 unit = {1.0 / ENERGY_UNITS_PER_HARTREE:.1f} Hartree")
    print(f"{args.trials} frames x {total:,} particles -> {args.output_dir}")


if __name__ == "__main__":
    main()
