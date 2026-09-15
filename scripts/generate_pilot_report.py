# ruff: noqa: E501
"""Generate comprehensive statistical tables, markdown summaries, and VR export for the pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantumflow.orchard_export import (
    create_orchard_bundle,
    export_trajectory_tape,
    generate_webxr_gallery_html,
)
from quantumflow.ot_cfm import load_model_params


def generate_reports_and_gallery(
    pilot_dir: Path = Path("outputs/transport/pilot"),
    analysis_file: Path | None = None,
) -> None:
    pilot_dir = Path(pilot_dir)
    analysis_path = analysis_file or (pilot_dir / "pilot_analysis.json")

    if not analysis_path.exists():
        print(f"Error: {analysis_path} not found.")
        return

    with open(analysis_path, encoding="utf-8") as f:
        analysis = json.load(f)

    summaries = analysis["summary_by_condition"]
    promoted = analysis.get("promoted_candidates", [])

    # 1. Main comparison table (BS=256)
    md_lines = []
    md_lines.append("# Isotropic-Hessian OT-CFM Multi-Seed Beta Pilot Results\n")
    md_lines.append("## Executive Summary")
    md_lines.append(
        "Evaluation of isotropic-Hessian penalty $\\beta$ across 10 values, 3 seeds ($N=2048$ fixed evaluation points, 2,000 steps), "
        "and batch-size ablation (128 vs 256).\n"
    )

    md_lines.append("## Main Benchmark Table (Batch Size = 256, 2,000 Steps, 3 Seeds)")
    header = (
        "| $\\beta$ | Empirical $W_2$ | $\\Delta W_2$ vs $\\beta=0$ (95% CI) | Sliced $W_2$ | "
        "Mean Eig Spread | Missing Modes | Mode Entropy | Wall Time (s) |"
    )
    separator = "|---|---|---|---|---|---|---|---|"
    md_lines.append(header)
    md_lines.append(separator)

    # Sort conditions by beta
    bs256_conds = [
        (k, v) for k, v in summaries.items() if v["batch_size"] == 256
    ]
    bs256_conds.sort(key=lambda x: x[1]["beta"])

    for cond_name, s in bs256_conds:
        b_val = s["beta"]
        b_str = f"{b_val:g}"
        w2_mean = s["empirical_w2"]["mean"]
        w2_std = s["empirical_w2"]["std"]
        
        delta_w2 = s["empirical_w2"]["paired_delta_mean"]
        ci_low, ci_high = s["empirical_w2"]["paired_delta_ci_95"]
        delta_str = f"{delta_w2:+.4f} [{ci_low:+.4f}, {ci_high:+.4f}]" if b_val > 0 else "0.0000 (ref)"
        
        sw2_mean = s["sliced_w2"]["mean"]
        sw2_std = s["sliced_w2"]["std"]
        
        eig_mean = s["mean_eigenvalue_spread"]["mean"]
        eig_std = s["mean_eigenvalue_spread"]["std"]
        
        missing = s["missing_modes"]["total"]
        entropy = s["mode_entropy"]["mean"]
        time_s = s["elapsed_seconds"]["mean"]

        row = (
            f"| `{b_str}` | {w2_mean:.4f} ± {w2_std:.4f} | {delta_str} | "
            f"{sw2_mean:.4f} ± {sw2_std:.4f} | {eig_mean:.4f} ± {eig_std:.4f} | "
            f"{missing}/24 | {entropy:.3f} | {time_s:.1f}s |"
        )
        md_lines.append(row)

    # 2. Pareto Step Count Table
    md_lines.append("\n## Integration Step Pareto Analysis (Sliced $W_2$ vs RK4 Steps)")
    p_header = "| $\\beta$ | 5 steps | 10 steps | 20 steps | 50 steps | 100 steps |"
    p_sep = "|---|---|---|---|---|---|"
    md_lines.append(p_header)
    md_lines.append(p_sep)

    for cond_name, s in bs256_conds:
        b_str = f"{s['beta']:g}"
        p = s["pareto_sliced_w2"]
        md_lines.append(
            f"| `{b_str}` | {p['5']:.4f} | {p['10']:.4f} | {p['20']:.4f} | "
            f"{p['50']:.4f} | {p['100']:.4f} |"
        )

    # 3. Batch-Size Ablation Table (128 vs 256)
    md_lines.append("\n## Batch Size Ablation (Batch Size 128 vs 256)")
    ab_header = "| $\\beta$ | BS | Empirical $W_2$ | Sliced $W_2$ (50s) | Sliced $W_2$ (5s) | Mean Eig Spread |"
    ab_sep = "|---|---|---|---|---|---|"
    md_lines.append(ab_header)
    md_lines.append(ab_sep)

    for b in [0.0, 1e-3, 1e-2]:
        for bs in [128, 256]:
            cond_key = f"bs{bs}_beta_{b:g}".replace(".", "_")
            if cond_key in summaries:
                s = summaries[cond_key]
                w2 = s["empirical_w2"]["mean"]
                sw2_50 = s["sliced_w2"]["mean"]
                sw2_5 = s["pareto_sliced_w2"]["5"]
                eig = s["mean_eigenvalue_spread"]["mean"]
                md_lines.append(
                    f"| `{b:g}` | {bs} | {w2:.4f} | {sw2_50:.4f} | {sw2_5:.4f} | {eig:.4f} |"
                )

    # 4. Candidate Promotion
    md_lines.append("\n## Candidate Promotion for Production Run")
    if promoted:
        for idx, c in enumerate(promoted, 1):
            md_lines.append(
                f"{idx}. **$\\beta = {c['beta']:g}$**:\n"
                f"   - Paired $\\Delta W_2$: {c['delta_w2']:+.4f} (95% CI: [{c['delta_w2_ci'][0]:+.4f}, {c['delta_w2_ci'][1]:+.4f}])\n"
                f"   - Low-step (5 steps) Sliced $W_2$ change: {c['delta_low_step_sw2']:+.4f}\n"
                f"   - Mean Hessian Eigenvalue Spread: {c['mean_eig_spread']:.4f} ($\\Delta$: {c['delta_eig_spread']:+.4f})\n"
                f"   - Missing modes across all seeds: {c['missing_modes']}.\n"
            )
    else:
        md_lines.append("No non-zero beta significantly outperformed beta=0 across metrics.\n")

    report_content = "\n".join(md_lines)
    report_file = pilot_dir / "PILOT_REPORT.md"
    report_file.write_text(report_content, encoding="utf-8")
    print(f"\nReport written to: {report_file}")
    print("\n" + report_content)

    raw_path = pilot_dir / "pilot_raw_results.json"
    if promoted and raw_path.exists():
        raw_results = json.loads(raw_path.read_text(encoding="utf-8"))
        promoted_beta = promoted[0]["beta"]
        eligible = [
            (name, metrics)
            for name, metrics in raw_results.items()
            if metrics["beta"] == promoted_beta and metrics["batch_size"] == 256
        ]
        best_name, _ = min(
            eligible,
            key=lambda item: item[1]["empirical_wasserstein_distance"],
        )
        model_path = pilot_dir / best_name / "model.npz"
        if model_path.exists():
            gallery_dir = pilot_dir / "gallery"
            tape_dir = gallery_dir / "tape"
            params = load_model_params(model_path)
            export_trajectory_tape(
                params,
                tape_dir,
                seed=1_000_003,
                title=f"OT-CFM beta={promoted_beta:g}: promoted pilot flow",
            )
            generate_webxr_gallery_html(
                gallery_dir / "index.html",
                particles_json_rel_path="tape/webxr_particles.json",
                title=f"QuantumFlow beta={promoted_beta:g}",
            )
            bundle_dir = create_orchard_bundle(
                tape_dir,
                gallery_dir / "bundles",
                title=f"QuantumFlow OT-CFM beta={promoted_beta:g}",
            )
            gallery_manifest = {
                "schema": "quantumflow/gallery/1",
                "source_run": best_name,
                "promoted_beta": promoted_beta,
                "webxr": "index.html",
                "tape": "tape",
                "orchard_bundle": str(bundle_dir) if bundle_dir else None,
            }
            (gallery_dir / "gallery.json").write_text(
                json.dumps(gallery_manifest, indent=2),
                encoding="utf-8",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pilot reports and VR gallery.")
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=Path("outputs/transport/pilot"),
        help="Pilot results directory",
    )
    args = parser.parse_args()
    generate_reports_and_gallery(pilot_dir=args.pilot_dir)


if __name__ == "__main__":
    main()
