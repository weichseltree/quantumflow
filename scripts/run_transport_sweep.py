"""Sweep runner across isotropic penalty weights beta."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_transport import train_and_eval


def run_sweep(
    betas: list[float] | None = None,
    steps: int = 1000,
    batch_size: int = 256,
    base_output_dir: Path | None = None,
) -> dict[str, dict]:
    if betas is None:
        betas = [0.0, 0.001, 0.01, 0.1, 1.0]

    sweep_results = {}
    base_dir = base_output_dir or Path("outputs/transport")
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STARTING ISOTROPIC-HESSIAN OT-CFM EXPERIMENT SWEEP")
    print(f"Betas: {betas}")
    print(f"Steps per run: {steps}")
    print("=" * 70)

    for beta in betas:
        run_name = f"beta_{beta:g}".replace(".", "_")
        run_dir = base_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n---> Running beta = {beta} (output: {run_dir})")
        res = train_and_eval(
            beta=beta,
            num_steps=steps,
            batch_size=batch_size,
            output_dir=run_dir,
        )
        sweep_results[str(beta)] = res

    # Save aggregated summary
    summary_path = base_dir / "sweep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSweep summary saved to: {summary_path}")

    # Print summary Pareto table
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS: PARETO TABLE & EIGENVALUE SPREAD")
    print("=" * 70)
    header = f"{'Beta':<10} | {'W2 Dist (50 steps)':<20} | {'Mean Eig Spread':<18} | {'Loss CFM':<10} | {'Loss Iso':<10}"
    print(header)
    print("-" * len(header))
    for beta_str, r in sweep_results.items():
        w2 = r["w2_distance"]
        eig_spr = r["mean_eigenvalue_spread"]
        l_cfm = r["final_loss_cfm"]
        l_iso = r["final_loss_iso"]
        print(f"{beta_str:<10} | {w2:<20.5f} | {eig_spr:<18.5f} | {l_cfm:<10.5f} | {l_iso:<10.5f}")
    print("=" * 70)

    return sweep_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep isotropic penalty weights beta.")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps per beta")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/transport"), help="Base output directory")
    args = parser.parse_args()

    run_sweep(steps=args.steps, batch_size=args.batch_size, base_output_dir=args.output_dir)


if __name__ == "__main__":
    main()
