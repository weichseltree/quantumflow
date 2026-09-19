"""Aggregate, validate, and report the multi-dimensional convex functional sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quantumflow import expdash


def collect_sweep_metrics(output_dir: Path) -> list[dict[str, Any]]:
    """Find and load all metric JSON files in the sweep output directory."""
    results = []
    for path in sorted(output_dir.glob("**/metrics_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["source_file"] = str(path.relative_to(output_dir))
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to read {path}: {e}")
    return results


def summarize_sweep(output_dir: Path) -> dict[str, Any]:
    metrics_list = collect_sweep_metrics(output_dir)
    if not metrics_list:
        print(f"No metrics found in {output_dir}")
        return {}

    print("\n" + "=" * 95)
    print("=== MULTI-DIMENSIONAL CONVEX POTENTIAL SWEEP SUMMARY ===")
    print(f"{'Run':<30} | {'Dim':<4} | {'Type':<18} | {'T MAE (Ha)':<11} | {'V MAE (Ha)':<11} | {'Orb MAE (Ha)':<12}")
    print("-" * 95)

    for item in metrics_list:
        run_name = Path(item.get("source_file", "")).parent.name or item.get("potential_type", "unknown")
        dim = f"{item.get('dimension')}D"
        p_type = item.get("potential_type", "")
        p_shape = item.get("potential_kwargs", {}).get("shape") or item.get("potential_kwargs", {}).get("mode", "")
        type_str = f"{p_type}:{p_shape}" if p_shape else p_type
        t_mae = f"{item.get('kinetic_energy_mae', 0.0):.4f}"
        v_mae = f"{item.get('potential_mae', 0.0):.4f}"
        orb_mae = f"{item.get('mean_orbital_energy_mae', 0.0):.4f}"
        print(f"{run_name:<30} | {dim:<4} | {type_str:<18} | {t_mae:<11} | {v_mae:<11} | {orb_mae:<12}")

    print("=" * 95 + "\n")

    summary = {
        "total_runs": len(metrics_list),
        "metrics": metrics_list,
    }

    summary_file = output_dir / "sweep_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved sweep summary to {summary_file}")

    # Report to ExpDash
    expdash.report(
        step=len(metrics_list),
        total=len(metrics_list),
        total_runs=len(metrics_list),
        status="completed",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize convex multidim sweep.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/convex_multidim/sweep"),
        help="Base directory containing sweep outputs",
    )
    args = parser.parse_args()
    summarize_sweep(args.output_dir)


if __name__ == "__main__":
    main()
