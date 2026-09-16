"""Wait for, validate, analyze, and publish the complete beta pilot."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

try:
    from .generate_pilot_report import generate_reports_and_gallery
    from .run_beta_pilot import (
        DEFAULT_BETAS,
        DEFAULT_SEEDS,
        DEFAULT_STEPS,
        PilotValidationError,
        analyze_pilot_results,
        collect_pilot_results,
        save_analysis,
        validate_pilot_results,
    )
except ImportError:  # Direct script execution.
    from generate_pilot_report import generate_reports_and_gallery
    from run_beta_pilot import (
        DEFAULT_BETAS,
        DEFAULT_SEEDS,
        DEFAULT_STEPS,
        PilotValidationError,
        analyze_pilot_results,
        collect_pilot_results,
        save_analysis,
        validate_pilot_results,
    )


def _matrix_state(
    raw_results: dict[str, dict[str, Any]],
    expected_steps: int,
) -> tuple[bool, str]:
    try:
        validate_pilot_results(raw_results, expected_steps=expected_steps)
    except PilotValidationError as exc:
        message = str(exc)
        if message.startswith("Pilot matrix mismatch:") and "unexpected=" not in message:
            return False, message
        raise
    return True, "complete"


def wait_for_complete_metrics(
    pilot_dir: Path,
    *,
    timeout_seconds: float,
    poll_seconds: float,
    expected_steps: int = DEFAULT_STEPS,
) -> dict[str, dict[str, Any]]:
    """Poll on CPU for the full matrix; malformed, duplicate, or unexpected runs fail."""
    if timeout_seconds < 0 or poll_seconds <= 0:
        raise ValueError("timeout_seconds must be non-negative and poll_seconds positive")
    deadline = time.monotonic() + timeout_seconds
    last_diagnostic = "no metrics discovered"
    while True:
        raw_results = collect_pilot_results(pilot_dir, require_any=False)
        complete, diagnostic = _matrix_state(raw_results, expected_steps)
        if complete:
            return raw_results
        last_diagnostic = diagnostic
        if time.monotonic() >= deadline:
            log_diagnostics = []
            for run_dir in sorted(pilot_dir.glob("pilot_*")):
                if (run_dir / "metrics.json").is_file():
                    continue
                log_path = run_dir / "train.log"
                if not log_path.is_file():
                    log_diagnostics.append(f"{run_dir.name}: metrics and train.log missing")
                    continue
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError as exc:
                    log_diagnostics.append(f"{run_dir.name}: cannot read train.log: {exc}")
                else:
                    tail = " | ".join(lines[-3:]) if lines else "<empty>"
                    log_diagnostics.append(f"{run_dir.name}: {tail}")
            log_note = (
                f"; incomplete_run_logs={log_diagnostics}"
                if log_diagnostics
                else "; no incomplete run directories found"
            )
            raise TimeoutError(
                f"Timed out after {timeout_seconds:g}s waiting for pilot metrics: "
                f"{last_diagnostic}{log_note}"
            )
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def finalize_pilot(
    pilot_dir: Path,
    *,
    timeout_seconds: float = 6 * 60 * 60,
    poll_seconds: float = 30,
    expected_steps: int = DEFAULT_STEPS,
    analyze_only: bool = False,
) -> dict[str, Any]:
    """Finalize a complete matrix; analyze-only performs no waiting or artifact export."""
    if analyze_only:
        raw_results = collect_pilot_results(pilot_dir)
    else:
        raw_results = wait_for_complete_metrics(
            pilot_dir,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            expected_steps=expected_steps,
        )
    analysis = analyze_pilot_results(
        raw_results,
        DEFAULT_BETAS,
        DEFAULT_SEEDS,
        expected_steps=expected_steps,
    )
    save_analysis(pilot_dir, raw_results, analysis)
    if not analyze_only:
        generate_reports_and_gallery(pilot_dir)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=Path("outputs/transport/pilot"))
    parser.add_argument("--timeout-seconds", type=float, default=6 * 60 * 60)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Validate and aggregate immediately; do not wait, report, or export a gallery",
    )
    args = parser.parse_args()
    finalize_pilot(
        args.pilot_dir,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
        expected_steps=args.steps,
        analyze_only=args.analyze_only,
    )


if __name__ == "__main__":
    main()
