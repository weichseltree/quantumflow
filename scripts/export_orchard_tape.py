"""Write the hall-sized OT-CFM flow tape for The Flow.

A thin front end onto :func:`quantumflow.orchard_export.export_trajectory_tape`.
The gallery tapes already on disk were written at the pilot's defaults, 512
particles over 60 steps, which is a thumbnail next to a room a visitor stands
inside; this runs the same model at hall scale.

Only one flow tape hangs. The learned potential carries an arbitrary
time-dependent additive gauge, so the vertical coordinate of a tape from one
trained model has no relation to that of a tape from another: two hung side by
side would invite a height comparison that does not exist. The beta comparison
belongs in a chart, where the axes are real. The gauge caveat travels in the
tape's own meta and onto the room's wall.

CPU, deliberately. The integration is a few thousand particles through a
three-layer MLP, and a JAX import that grabbed a CUDA context would contend
invisibly with whatever holds the lane.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

from quantumflow.orchard_export import export_trajectory_tape  # noqa: E402
from quantumflow.ot_cfm import load_model_params  # noqa: E402

#: The pilot promoted no beta -- every non-zero strength cost accuracy, and the
#: analysis retained the baseline for the representative artefact. So the tape
#: that hangs is the baseline, and that is a finding rather than a convenience.
DEFAULT_MODEL = Path("outputs/transport/pilot/pilot_beta_0_s0_bs256/model.npz")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument(
        "--steps",
        type=int,
        default=240,
        help=(
            "RK4 integration steps. Well past the step-count knee the pilot's Pareto "
            "analysis measured, so the motion a visitor scrubs is integration-exact "
            "rather than step-limited."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tape/flow"))
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"no model at {args.model}")

    params = load_model_params(args.model)
    print(f"model {args.model}")
    print(f"{args.particles:,} particles over {args.steps} RK4 steps")

    written = export_trajectory_tape(
        params,
        args.output_dir,
        num_particles=args.particles,
        num_steps=args.steps,
        seed=args.seed,
        title="Isotropic-Hessian OT-CFM particle flow, the retained baseline",
        provenance={
            "synthetic": False,
            "condition": "beta = 0, the pilot's retained baseline",
            "why_this_model": (
                "no non-zero beta passed the preregistered promotion gates; the analysis "
                "retained the baseline for the representative artefact"
            ),
            "beta_comparison": (
                "hangs as a chart, not as a second tape: Phi's additive gauge makes two "
                "independently trained flows incomparable by height"
            ),
        },
        source_run=args.model.parent.name,
        source_model=str(args.model),
        git_sha=git_sha(),
    )
    print(f"-> {written}")


if __name__ == "__main__":
    main()
