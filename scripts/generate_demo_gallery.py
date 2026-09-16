"""Train a tiny OT-CFM model and export a demo 2D/3D/WebXR/Orchard gallery.

This exercises the same export pipeline as the full beta pilot's
representative gallery (see ``scripts/generate_pilot_report.py``) without
running the 39-run pilot: a small model trains in seconds on a laptop CPU,
then the trajectory is exported as a ``video/tape/1`` tape, an offline WebXR
viewer, and an ``orchard/bundle/1`` bundle.

Building the tape/viewer/bundle requires the optional Orchard stack (see the
"Quick gallery demo" section of ``README.md``). Training and saving the
model does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .run_transport import train_and_eval
except ImportError:  # Direct script execution.
    from run_transport import train_and_eval


def generate_demo_gallery(
    output_dir: Path,
    *,
    beta: float = 0.0,
    steps: int = 300,
    batch_size: int = 64,
    seed: int = 42,
    eval_samples: int = 512,
    num_particles: int = 256,
    tape_steps: int = 40,
    skip_train: bool = False,
    skip_bundle: bool = False,
) -> dict[str, Path]:
    """Train a tiny model (unless skipped) and export a demo gallery.

    Returns the paths of every artifact written. Raises if Orchard is not
    installed and the gallery step is requested (``skip_bundle`` only skips
    the bundle; tape and WebXR export still require ``orchard_tape``).
    """
    output_dir = Path(output_dir)
    model_path = output_dir / "model.npz"

    if not skip_train:
        train_and_eval(
            beta=beta,
            num_steps=steps,
            batch_size=batch_size,
            seed=seed,
            output_dir=output_dir,
            eval_samples=eval_samples,
        )
    elif not model_path.is_file():
        raise FileNotFoundError(f"--skip-train requires an existing model at {model_path}")

    from quantumflow.orchard_export import (
        create_orchard_bundle,
        export_trajectory_tape,
        generate_webxr_gallery_html,
    )
    from quantumflow.ot_cfm import load_model_params

    params = load_model_params(model_path)
    gallery_dir = output_dir / "gallery"
    tape_dir = gallery_dir / "tape"
    title = f"QuantumFlow demo beta={beta:g}"

    export_trajectory_tape(
        params,
        tape_dir,
        num_particles=num_particles,
        num_steps=tape_steps,
        seed=seed + 1,
        title=title,
        source_run=output_dir.name,
        source_model="model.npz",
        provenance={"demo": True, "purpose": "quick local gallery demo"},
    )
    generate_webxr_gallery_html(
        gallery_dir / "index.html",
        particles_json_rel_path="tape/webxr_particles.json",
        title=title,
    )

    result_paths: dict[str, Path] = {
        "model": model_path,
        "tape_dir": tape_dir,
        "webxr_html": gallery_dir / "index.html",
    }

    manifest: dict[str, object] = {
        "schema": "quantumflow/gallery/2",
        "source_run": output_dir.name,
        "source_model": "model.npz",
        "beta": beta,
        "webxr": "index.html",
        "tape": "tape",
    }
    if not skip_bundle:
        bundle_dir = Path(create_orchard_bundle(tape_dir, gallery_dir / "bundles", title=title))
        result_paths["orchard_bundle"] = bundle_dir
        manifest["orchard_bundle"] = str(bundle_dir.relative_to(gallery_dir))

    manifest_path = gallery_dir / "gallery.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    result_paths["gallery_manifest"] = manifest_path
    return result_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/demo_gallery/demo"),
        help="Directory for the trained model and gallery/ (default: %(default)s)",
    )
    parser.add_argument("--beta", type=float, default=0.0, help="Isotropic-Hessian penalty weight")
    parser.add_argument("--steps", type=int, default=300, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--seed", type=int, default=42, help="Experiment seed")
    parser.add_argument("--eval-samples", type=int, default=512, help="Evaluation sample count")
    parser.add_argument(
        "--particles", type=int, default=256, help="Particle count exported to the tape"
    )
    parser.add_argument(
        "--tape-steps", type=int, default=40, help="RK4 integration steps exported to the tape"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse an existing model.npz in --output-dir instead of training",
    )
    parser.add_argument(
        "--no-bundle",
        action="store_true",
        help="Skip creating the orchard/bundle/1 bundle (tape + WebXR viewer only)",
    )
    args = parser.parse_args()

    paths = generate_demo_gallery(
        args.output_dir,
        beta=args.beta,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        eval_samples=args.eval_samples,
        num_particles=args.particles,
        tape_steps=args.tape_steps,
        skip_train=args.skip_train,
        skip_bundle=args.no_bundle,
    )

    print("\nDemo gallery ready:")
    for name, path in paths.items():
        print(f"  {name}: {path.resolve()}")
    gallery_dir = paths["webxr_html"].parent
    print(
        "\nServe it locally with:\n"
        f"  python -m http.server --directory {gallery_dir} 8000\n"
        "then open http://localhost:8000/"
    )


if __name__ == "__main__":
    main()
