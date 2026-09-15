"""Aggregate completed beta-pilot runs and build reports and gallery artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from generate_pilot_report import generate_reports_and_gallery
from run_beta_pilot import (
    DEFAULT_BETAS,
    DEFAULT_SEEDS,
    analyze_pilot_results,
    collect_pilot_results,
)


def main() -> None:
    pilot_dir = Path("outputs/transport/pilot")
    raw_results = collect_pilot_results(pilot_dir)
    expected_runs = len(DEFAULT_BETAS) * len(DEFAULT_SEEDS) + 3 * len(DEFAULT_SEEDS)
    if len(raw_results) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} completed runs, found {len(raw_results)}"
        )

    analysis = analyze_pilot_results(raw_results, DEFAULT_BETAS, DEFAULT_SEEDS)
    (pilot_dir / "pilot_raw_results.json").write_text(
        json.dumps(raw_results, indent=2),
        encoding="utf-8",
    )
    (pilot_dir / "pilot_analysis.json").write_text(
        json.dumps(analysis, indent=2),
        encoding="utf-8",
    )
    generate_reports_and_gallery(pilot_dir)


if __name__ == "__main__":
    main()
