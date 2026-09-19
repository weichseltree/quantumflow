"""Tests for sweep finalizers: multidim convex and 3D transport."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.finalize_convex_multidim_sweep import summarize_sweep as summarize_convex_sweep
from scripts.finalize_transport_3d_sweep import summarize_3d_sweep


def test_finalize_convex_multidim_sweep(tmp_path: Path) -> None:
    """Test aggregation and reporting for multidim convex sweep."""
    run1_dir = tmp_path / "run_3d_pw_box"
    run1_dir.mkdir(parents=True)
    metrics1 = {
        "dimension": 3,
        "potential_type": "piecewise_constant",
        "potential_kwargs": {"shape": "box"},
        "kinetic_energy_mae": 1400.0,
        "potential_mae": 80.0,
        "mean_orbital_energy_mae": 25.0,
    }
    with open(run1_dir / "metrics_3d.json", "w", encoding="utf-8") as f:
        json.dump(metrics1, f)

    run2_dir = tmp_path / "run_2d_gaussian"
    run2_dir.mkdir(parents=True)
    metrics2 = {
        "dimension": 2,
        "potential_type": "gaussian",
        "potential_kwargs": {},
        "kinetic_energy_mae": 250.0,
        "potential_mae": 230.0,
        "mean_orbital_energy_mae": 20.0,
    }
    with open(run2_dir / "metrics_2d.json", "w", encoding="utf-8") as f:
        json.dump(metrics2, f)

    summary = summarize_convex_sweep(tmp_path)
    assert summary["total_runs"] == 2
    assert (tmp_path / "sweep_summary.json").is_file()


def test_finalize_transport_3d_sweep(tmp_path: Path) -> None:
    """Test aggregation, paired delta calculation, and report generation for 3D transport sweep."""
    for b in [0.0, 0.5]:
        for s in [42, 43]:
            run_dir = tmp_path / f"transport-3d-b{b}-s{s}"
            run_dir.mkdir(parents=True)
            m = {
                "schema_version": 2,
                "seed": s,
                "dimension": 3,
                "beta": b,
                "final_loss": 0.2 + b * 0.1,
                "final_loss_cfm": 0.2,
                "final_loss_iso": b * 0.1,
                "sliced_wasserstein_distance": 0.25 + b * 0.05,
                "empirical_wasserstein_distance": 0.8 + b * 0.1,
                "mean_eigenvalue_spread": 0.35 - b * 0.05,
                "max_eigenvalue_spread": 0.9,
                "modes_covered": 8,
                "missing_modes": 0,
                "mode_counts": [128] * 8,
                "mode_entropy": 0.99,
                "pareto_step_sliced_wasserstein_distance": {"5": 0.26, "50": 0.25},
                "pareto_step_empirical_wasserstein_distance": {"5": 0.81, "50": 0.80},
                "training_seconds": 20.0,
                "evaluation_seconds": 30.0,
                "elapsed_seconds": 50.0,
            }
            with open(run_dir / "metrics_3d.json", "w", encoding="utf-8") as f:
                json.dump(m, f)

    summary = summarize_3d_sweep(tmp_path)
    assert summary["total_runs"] == 4
    assert (tmp_path / "sweep_summary.json").is_file()
    assert (tmp_path / "TRANSPORT_3D_REPORT.md").is_file()

    report_content = (tmp_path / "TRANSPORT_3D_REPORT.md").read_text()
    assert "3D Isotropic-Hessian OT-CFM Sweep Report" in report_content
    assert "0.50" in report_content
