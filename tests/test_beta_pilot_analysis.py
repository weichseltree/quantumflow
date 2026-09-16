# ruff: noqa: I001

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from scripts.finalize_beta_pilot import wait_for_complete_metrics
from scripts.generate_pilot_report import _representative_run, render_report
from scripts.run_beta_pilot import (
    PilotValidationError,
    analyze_pilot_results,
    collect_pilot_results,
)


BETAS = [0.0, 0.01]
SEEDS = [0, 1, 2]
STEPS = ("5", "10", "20", "50", "100")


def metric(beta: float, seed: int, endpoint_delta: float = 0.0) -> dict:
    baseline = 1.0 + seed * 0.1
    endpoint = baseline + (endpoint_delta if beta else 0.0)
    pareto = {step: endpoint + 0.5 / int(step) for step in STEPS}
    return {
        "schema_version": 2,
        "seed": seed,
        "config": {
            "beta": beta,
            "num_steps": 2000,
            "batch_size": 256,
            "learning_rate": 0.001,
            "eval_samples": 2048,
            "ode_step_counts": [5, 10, 20, 50, 100],
            "fixed_eval_seed": 1_000_003,
            "step_offset": 0,
            "sample_offset": 0,
        },
        "beta": beta,
        "batch_size": 256,
        "num_steps": 2000,
        "start_step": 0,
        "end_step": 2000,
        "start_samples": 0,
        "end_samples": 512_000,
        "empirical_wasserstein_distance": endpoint,
        "sliced_wasserstein_distance": endpoint / 2,
        "mean_eigenvalue_spread": 0.4,
        "max_eigenvalue_spread": 0.8,
        "modes_covered": 8,
        "missing_modes": 0,
        "mode_counts": [256] * 8,
        "mode_entropy": 2.0,
        "final_loss": 0.3,
        "final_loss_cfm": 0.2,
        "final_loss_iso": 0.1,
        "training_seconds": 8.0,
        "evaluation_seconds": 2.0,
        "elapsed_seconds": 10.0,
        "pareto_step_sliced_wasserstein_distance": {
            step: value / 2 for step, value in pareto.items()
        },
        "pareto_step_empirical_wasserstein_distance": pareto,
        "ode_function_evaluations": {step: 4 * int(step) for step in STEPS},
        "evaluation_seed": 1_000_003,
        "projection_seed": 2_000_003,
    }


def complete_results(delta: float) -> dict[str, dict]:
    return {
        f"actual-directory-{beta}-{seed}": metric(beta, seed, delta)
        for beta in BETAS
        for seed in SEEDS
    }


def analyze(results: dict[str, dict]) -> dict:
    return analyze_pilot_results(
        results,
        BETAS,
        SEEDS,
        expected_steps=2000,
        include_ablation=False,
    )


def test_analysis_promotes_only_conservative_non_sacrifice_candidate() -> None:
    analysis = analyze(complete_results(-0.2))

    assert [candidate["beta"] for candidate in analysis["promoted_candidates"]] == [0.01]
    candidate = analysis["promoted_candidates"][0]
    assert candidate["paired_endpoint_interval"]["resamples"] == 27
    assert candidate["paired_endpoint_interval"]["upper"] < 0
    assert candidate["delta_w2_by_seed"] == {
        "0": pytest.approx(-0.2),
        "1": pytest.approx(-0.2),
        "2": pytest.approx(-0.2),
    }
    assert set(candidate["delta_low_step_empirical_w2_by_seed"]) == {"5", "10", "20"}
    assert analysis["inference"]["n_pairs"] == 3
    assert len(analysis["raw_runs"]) == 6
    assert isinstance(
        analysis["summary_by_condition"]["bs256_beta_0"]["pareto_empirical_w2"]["5"],
        float,
    )


def test_analysis_allows_no_promotions_when_candidate_is_worse() -> None:
    analysis = analyze(complete_results(0.1))

    assert analysis["promoted_candidates"] == []
    assert "endpoint_empirical_w2_sacrifice" in analysis["all_candidates"][0]["promotion_failures"]
    report = render_report(analysis)
    assert "no beta is promoted" in report
    assert "not adjusted for multiple comparisons" in report


def test_missing_pareto_metric_is_rejected_instead_of_defaulting_to_zero() -> None:
    results = complete_results(-0.2)
    del results["actual-directory-0.01-1"]["pareto_step_empirical_wasserstein_distance"]["5"]

    with pytest.raises(PilotValidationError, match="must have exactly steps"):
        analyze(results)


def test_matrix_validation_uses_metadata_and_rejects_duplicate_and_missing() -> None:
    results = complete_results(-0.2)
    results["pilot_beta_1_0_s9_bs256"] = results.pop("actual-directory-0.01-2")
    analyze(results)

    results["duplicate-name"] = dict(results["pilot_beta_1_0_s9_bs256"])
    with pytest.raises(PilotValidationError, match="Duplicate condition"):
        analyze(results)

    del results["duplicate-name"]
    del results["actual-directory-0.01-1"]
    with pytest.raises(PilotValidationError, match="missing=.*seed=1"):
        analyze(results)


def test_same_count_wrong_condition_is_rejected_as_missing_and_unexpected() -> None:
    results = complete_results(-0.2)
    results.pop("actual-directory-0.01-1")
    results["unexpected-condition"] = metric(0.02, 1, -0.2)

    with pytest.raises(PilotValidationError, match=r"missing=.*unexpected="):
        analyze(results)


def test_inconsistent_budget_and_nonfinite_values_are_rejected() -> None:
    results = complete_results(-0.2)
    results["actual-directory-0.01-0"]["end_samples"] = 511_999
    with pytest.raises(PilotValidationError, match="end_samples"):
        analyze(results)

    results = complete_results(-0.2)
    results["actual-directory-0.01-0"]["final_loss_iso"] = float("nan")
    with pytest.raises(PilotValidationError, match="must be finite"):
        analyze(results)


def test_representative_run_is_median_seed_not_best() -> None:
    analysis = analyze(complete_results(-0.2))
    run_name, seed = _representative_run(analysis, 0.01)

    assert seed == 1
    assert run_name == "actual-directory-0.01-1"

    baseline_analysis = analyze(complete_results(0.2))
    baseline_name, baseline_seed = _representative_run(baseline_analysis, 0.0)
    assert baseline_seed == 1
    assert baseline_name == "actual-directory-0.0-1"


def test_collector_and_waiter_do_not_claim_incomplete_matrix(tmp_path: Path) -> None:
    run_dir = tmp_path / "pilot_beta_0_s0_bs256"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(json.dumps(metric(0.0, 0)), encoding="utf-8")

    collected = collect_pilot_results(tmp_path)
    assert list(collected) == ["pilot_beta_0_s0_bs256"]
    with pytest.raises(TimeoutError, match="missing=.*beta=1"):
        wait_for_complete_metrics(
            tmp_path,
            timeout_seconds=0,
            poll_seconds=0.01,
            expected_steps=2000,
        )
