"""Generate the beta-pilot comparison report and representative gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _condition_rows(analysis: dict[str, Any], batch_size: int) -> list[dict[str, Any]]:
    rows = [
        summary
        for summary in analysis["summary_by_condition"].values()
        if summary["batch_size"] == batch_size
    ]
    return sorted(rows, key=lambda summary: summary["beta"])


def render_report(analysis: dict[str, Any]) -> str:
    """Render a concise, cautious comparison table from a validated aggregate."""
    lines = [
        "# Isotropic-Hessian OT-CFM beta pilot",
        "",
        (
            "Three paired training seeds per condition; values are mean +/- sample SD. "
            "Intervals enumerate all paired n-to-n bootstrap resamples. This is exploratory "
            "small-n inference, conditional on fixed evaluation data/projections and not "
            "adjusted for multiple comparisons."
        ),
        "",
        "| beta | endpoint empirical W2 | paired delta [95% interval] | "
        "5-step empirical W2 (paired delta) | missing modes |",
        "|---:|---:|---:|---:|---:|",
    ]
    for summary in _condition_rows(analysis, 256):
        endpoint = summary["empirical_w2"]
        low_step = summary["pareto_empirical_w2_statistics"]["5"]
        if summary["beta"] == 0:
            delta = "reference"
        else:
            lower, upper = endpoint["paired_delta_ci_95"]
            delta = f"{endpoint['paired_delta_mean']:+.4f} [{lower:+.4f}, {upper:+.4f}]"
        lines.append(
            f"| {summary['beta']:g} | {endpoint['mean']:.4f} +/- {endpoint['std']:.4f} | "
            f"{delta} | {low_step['mean']:.4f} "
            f"({low_step['paired_delta_mean']:+.4f}) | "
            f"{summary['missing_modes']['total']} |"
        )

    promoted = analysis["promoted_candidates"]
    lines.extend(["", "## Promotion decision", ""])
    if promoted:
        values = ", ".join(f"`{candidate['beta']:g}`" for candidate in promoted)
        lines.append(
            f"Promoted beta value(s): {values}. Each passed the preregistered endpoint, "
            "low-step empirical-W2, mode-coverage, and conservative interval gates."
        )
    else:
        lines.append(
            "No non-zero beta passed every preregistered gate; no beta is promoted. "
            "The baseline is retained for the representative artifact."
        )
    lines.extend(
        [
            "",
            "The machine-readable `pilot_analysis.json` contains labeled seed values and "
            "paired deltas for endpoint and every ODE step count, full raw metrics, source-run "
            "provenance, validation design, and the failure reasons for every candidate.",
            "",
        ]
    )
    return "\n".join(lines)


def _representative_run(
    analysis: dict[str, Any],
    beta: float,
) -> tuple[str, int]:
    condition_name = f"bs256_beta_{beta:g}".replace(".", "_")
    summary = analysis["summary_by_condition"][condition_name]
    values = summary["empirical_w2"]["values_by_seed"]
    ordered_seeds = sorted((float(value), int(seed)) for seed, value in values.items())
    median_seed = ordered_seeds[len(ordered_seeds) // 2][1]
    return summary["source_runs_by_seed"][str(median_seed)], median_seed


def export_representative_gallery(
    pilot_dir: Path,
    analysis: dict[str, Any],
) -> Path:
    """Export the median-seed promoted run, or the median-seed baseline if none passed."""
    from quantumflow.orchard_export import (
        create_orchard_bundle,
        export_trajectory_tape,
        generate_webxr_gallery_html,
    )
    from quantumflow.ot_cfm import load_model_params

    promoted = analysis["promoted_candidates"]
    beta = float(promoted[0]["beta"]) if promoted else 0.0
    selection = "promoted" if promoted else "baseline_no_promotion"
    run_name, seed = _representative_run(analysis, beta)
    model_path = pilot_dir / run_name / "model.npz"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Representative {selection} model is missing: {model_path}"
        )

    gallery_dir = pilot_dir / "gallery"
    tape_dir = gallery_dir / "tape"
    params = load_model_params(model_path)
    export_trajectory_tape(
        params,
        tape_dir,
        seed=1_000_003,
        title=f"OT-CFM beta={beta:g}: {selection} median-seed flow",
        source_run=run_name,
        source_model=str(model_path.relative_to(pilot_dir)),
        provenance={
            "analysis_schema": analysis["schema"],
            "selection": selection,
            "representative_seed": seed,
            "beta": beta,
        },
    )
    generate_webxr_gallery_html(
        gallery_dir / "index.html",
        particles_json_rel_path="tape/webxr_particles.json",
        title=f"QuantumFlow beta={beta:g}",
    )
    bundle_dir = create_orchard_bundle(
        tape_dir,
        gallery_dir / "bundles",
        title=f"QuantumFlow OT-CFM beta={beta:g}",
    )
    if bundle_dir is None:
        raise RuntimeError("Official orchard bundle exporter is unavailable")
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise RuntimeError(f"Orchard exporter returned a missing bundle: {bundle_dir}")

    manifest = {
        "schema": "quantumflow/gallery/2",
        "source_run": run_name,
        "source_model": str(model_path.relative_to(pilot_dir)),
        "representative_seed": seed,
        "selection": selection,
        "beta": beta,
        "webxr": "index.html",
        "tape": "tape",
        "orchard_bundle": str(bundle_dir.relative_to(gallery_dir)),
    }
    manifest_path = gallery_dir / "gallery.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest_path


def generate_reports_and_gallery(
    pilot_dir: Path = Path("outputs/transport/pilot"),
    analysis_file: Path | None = None,
    *,
    export_gallery: bool = True,
) -> dict[str, Path | None]:
    """Write the report first, then fail loudly if the requested gallery cannot be built."""
    pilot_dir = Path(pilot_dir)
    analysis_path = analysis_file or (pilot_dir / "pilot_analysis.json")
    if not analysis_path.is_file():
        raise FileNotFoundError(f"Pilot analysis not found: {analysis_path}")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("schema") != "quantumflow/beta-pilot-analysis/2":
        raise ValueError(f"Unsupported pilot analysis schema in {analysis_path}")

    report_path = pilot_dir / "PILOT_REPORT.md"
    report_path.write_text(render_report(analysis), encoding="utf-8")
    gallery_path = (
        export_representative_gallery(pilot_dir, analysis) if export_gallery else None
    )
    return {"report": report_path, "gallery_manifest": gallery_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=Path("outputs/transport/pilot"))
    parser.add_argument("--analysis-file", type=Path)
    parser.add_argument("--no-gallery", action="store_true")
    args = parser.parse_args()
    generate_reports_and_gallery(
        args.pilot_dir,
        args.analysis_file,
        export_gallery=not args.no_gallery,
    )


if __name__ == "__main__":
    main()
