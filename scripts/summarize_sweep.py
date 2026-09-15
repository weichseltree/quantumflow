"""Summarize sweep outputs into table and JSON summary."""

from __future__ import annotations

import json
from pathlib import Path


def summarize(base_dir: Path = Path("outputs/transport")) -> None:
    summary = {}
    betas = [0.0, 0.001, 0.01, 0.1, 1.0]

    print("=" * 85)
    print(f"{'Beta':<8} | {'W2 Dist (50s)':<15} | {'Mean Eig Spread':<16} | {'Final Loss':<12} | {'CFM Loss':<10} | {'Iso Loss':<10}")
    print("-" * 85)

    for beta in betas:
        run_name = f"beta_{beta:g}".replace(".", "_")
        metrics_file = base_dir / run_name / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, encoding="utf-8") as f:
                d = json.load(f)
            summary[str(beta)] = d
            w2 = d.get("w2_distance", 0.0)
            eig_spr = d.get("mean_eigenvalue_spread", 0.0)
            loss = d.get("final_loss", 0.0)
            l_cfm = d.get("final_loss_cfm", 0.0)
            l_iso = d.get("final_loss_iso", 0.0)
            print(f"{str(beta):<8} | {w2:<15.5f} | {eig_spr:<16.5f} | {loss:<12.5f} | {l_cfm:<10.5f} | {l_iso:<10.5f}")

    print("=" * 85)

    print("\n" + "=" * 65)
    print("PARETO ANALYSIS: Sliced Wasserstein vs ODE Integration Steps")
    print("=" * 65)
    print(f"{'Beta':<8} | {'5 steps':<10} | {'10 steps':<10} | {'20 steps':<10} | {'50 steps':<10} | {'100 steps':<10}")
    print("-" * 65)
    for beta_str, d in summary.items():
        p = d.get("pareto_step_w2", {})
        s5 = p.get("5", 0.0)
        s10 = p.get("10", 0.0)
        s20 = p.get("20", 0.0)
        s50 = p.get("50", 0.0)
        s100 = p.get("100", 0.0)
        print(f"{beta_str:<8} | {s5:<10.4f} | {s10:<10.4f} | {s20:<10.4f} | {s50:<10.4f} | {s100:<10.4f}")
    print("=" * 65)

    summary_file = base_dir / "sweep_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_file}")


if __name__ == "__main__":
    summarize()
