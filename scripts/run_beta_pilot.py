"""Run and analyze the multi-seed isotropic-Hessian OT-CFM beta pilot."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .run_transport import train_and_eval
except ImportError:  # Direct script execution.
    from run_transport import train_and_eval

DEFAULT_BETAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
DEFAULT_SEEDS = [0, 1, 2]
ABLATION_BETAS = [0.0, 1e-3, 1e-2]
DEFAULT_STEPS = 2000
STEP_COUNTS = ("5", "10", "20", "50", "100")
LOW_STEP_COUNTS = ("5", "10", "20")

SCALAR_METRICS = (
    "empirical_wasserstein_distance",
    "sliced_wasserstein_distance",
    "mean_eigenvalue_spread",
    "max_eigenvalue_spread",
    "modes_covered",
    "missing_modes",
    "mode_entropy",
    "final_loss",
    "final_loss_cfm",
    "final_loss_iso",
    "training_seconds",
    "evaluation_seconds",
    "elapsed_seconds",
)


class PilotValidationError(ValueError):
    """Pilot inputs do not form the preregistered, complete experiment matrix."""


def exact_bootstrap_ci(
    deltas: list[float] | np.ndarray,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Return the percentile interval over every size-n bootstrap resample."""
    values = np.asarray(deltas, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("deltas must be a non-empty finite one-dimensional sequence")
    means = np.fromiter(
        (float(np.mean(sample)) for sample in itertools.product(values, repeat=values.size)),
        dtype=float,
        count=values.size**values.size,
    )
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def bootstrap_ci(
    deltas: list[float] | np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compatibility wrapper; inference uses deterministic exact resampling."""
    del n_boot, seed
    return exact_bootstrap_ci(deltas, ci=ci)


def _expected_matrix(
    betas: list[float],
    seeds: list[int],
    include_ablation: bool,
) -> set[tuple[int, float, int]]:
    expected = {(256, float(beta), int(seed)) for beta in betas for seed in seeds}
    if include_ablation:
        expected.update(
            (128, float(beta), int(seed)) for beta in ABLATION_BETAS for seed in seeds
        )
    return expected


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise PilotValidationError(f"{label} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise PilotValidationError(f"{label} must be finite, got {value!r}")
    return result


def _integer(value: Any, label: str) -> int:
    result = _finite_number(value, label)
    if not result.is_integer():
        raise PilotValidationError(f"{label} must be an integer, got {value!r}")
    return int(result)


def _validate_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _validate_finite_tree(nested, f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_finite_tree(nested, f"{label}[{index}]")
    elif isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        _finite_number(value, label)


def _run_identity(run_name: str, metrics: dict[str, Any]) -> tuple[int, float, int]:
    try:
        config = metrics["config"]
        beta = _finite_number(config["beta"], f"{run_name}.config.beta")
        batch_size = _integer(config["batch_size"], f"{run_name}.config.batch_size")
        seed = _integer(metrics["seed"], f"{run_name}.seed")
    except (KeyError, TypeError) as exc:
        raise PilotValidationError(f"{run_name}: missing run identity metadata: {exc}") from exc
    for key, expected in (("beta", beta), ("batch_size", batch_size)):
        actual = _finite_number(metrics.get(key), f"{run_name}.{key}")
        if actual != expected:
            raise PilotValidationError(
                f"{run_name}: top-level {key}={actual} disagrees with config {expected}"
            )
    return batch_size, beta, seed


def validate_pilot_results(
    raw_results: dict[str, dict[str, Any]],
    betas: list[float] | None = None,
    seeds: list[int] | None = None,
    *,
    expected_steps: int = DEFAULT_STEPS,
    include_ablation: bool = True,
) -> dict[tuple[int, float, int], tuple[str, dict[str, Any]]]:
    """Validate identities, budgets, metrics, and the exact preregistered matrix."""
    betas = DEFAULT_BETAS if betas is None else betas
    seeds = DEFAULT_SEEDS if seeds is None else seeds
    indexed: dict[tuple[int, float, int], tuple[str, dict[str, Any]]] = {}

    for run_name, metrics in raw_results.items():
        _validate_finite_tree(metrics, run_name)
        if metrics.get("schema_version") != 2:
            raise PilotValidationError(
                f"{run_name}.schema_version={metrics.get('schema_version')!r}; expected 2"
            )
        identity = _run_identity(run_name, metrics)
        if identity in indexed:
            other = indexed[identity][0]
            raise PilotValidationError(
                f"Duplicate condition bs={identity[0]}, beta={identity[1]:g}, "
                f"seed={identity[2]} in {other!r} and {run_name!r}"
            )
        indexed[identity] = (run_name, metrics)
        batch_size, _, _ = identity

        config = metrics["config"]
        budget_fields = {
            "config.num_steps": config.get("num_steps"),
            "config.step_offset": config.get("step_offset"),
            "config.sample_offset": config.get("sample_offset"),
            "num_steps": metrics.get("num_steps"),
            "start_step": metrics.get("start_step"),
            "end_step": metrics.get("end_step"),
            "start_samples": metrics.get("start_samples"),
            "end_samples": metrics.get("end_samples"),
        }
        expected_budget = {
            "config.num_steps": expected_steps,
            "config.step_offset": 0,
            "config.sample_offset": 0,
            "num_steps": expected_steps,
            "start_step": 0,
            "end_step": expected_steps,
            "start_samples": 0,
            "end_samples": expected_steps * batch_size,
        }
        for field, value in budget_fields.items():
            actual = _integer(value, f"{run_name}.{field}")
            if actual != expected_budget[field]:
                raise PilotValidationError(
                    f"{run_name}.{field}={actual:g}; expected {expected_budget[field]:g}"
                )

        for metric in SCALAR_METRICS:
            _finite_number(metrics.get(metric), f"{run_name}.{metric}")
        if int(metrics["modes_covered"]) + int(metrics["missing_modes"]) != 8:
            raise PilotValidationError(f"{run_name}: modes_covered + missing_modes must equal 8")
        mode_counts = metrics.get("mode_counts")
        if not isinstance(mode_counts, list) or len(mode_counts) != 8:
            raise PilotValidationError(f"{run_name}.mode_counts must contain exactly 8 values")
        for index, value in enumerate(mode_counts):
            count = _integer(value, f"{run_name}.mode_counts[{index}]")
            if count < 0:
                raise PilotValidationError(f"{run_name}.mode_counts[{index}] must be non-negative")
        assigned = sum(mode_counts)
        unassigned = _integer(
            metrics.get("unassigned_samples", 0), f"{run_name}.unassigned_samples"
        )
        if assigned + unassigned != config.get("eval_samples"):
            raise PilotValidationError(
                f"{run_name}.mode_counts plus unassigned_samples must sum to config.eval_samples"
            )

        configured_steps = tuple(str(step) for step in config.get("ode_step_counts", ()))
        if configured_steps != STEP_COUNTS:
            raise PilotValidationError(
                f"{run_name}.config.ode_step_counts={configured_steps}; expected {STEP_COUNTS}"
            )
        for metric in (
            "pareto_step_sliced_wasserstein_distance",
            "pareto_step_empirical_wasserstein_distance",
        ):
            values = metrics.get(metric)
            if not isinstance(values, dict) or set(values) != set(STEP_COUNTS):
                raise PilotValidationError(
                    f"{run_name}.{metric} must have exactly steps {list(STEP_COUNTS)}"
                )
            for step in STEP_COUNTS:
                _finite_number(values[step], f"{run_name}.{metric}.{step}")
        function_evaluations = metrics.get("ode_function_evaluations")
        if not isinstance(function_evaluations, dict) or set(function_evaluations) != set(
            STEP_COUNTS
        ):
            raise PilotValidationError(
                f"{run_name}.ode_function_evaluations must have exactly steps "
                f"{list(STEP_COUNTS)}"
            )
        for step in STEP_COUNTS:
            actual = _integer(
                function_evaluations[step],
                f"{run_name}.ode_function_evaluations.{step}",
            )
            if actual != 4 * int(step):
                raise PilotValidationError(
                    f"{run_name}.ode_function_evaluations.{step}={actual}; "
                    f"expected {4 * int(step)}"
                )

    expected = _expected_matrix(betas, seeds, include_ablation)
    actual = set(indexed)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={[_format_identity(item) for item in missing]}")
        if unexpected:
            details.append(f"unexpected={[_format_identity(item) for item in unexpected]}")
        raise PilotValidationError("Pilot matrix mismatch: " + "; ".join(details))

    eval_shapes = {
        (
            metrics["config"].get("eval_samples"),
            metrics["config"].get("fixed_eval_seed"),
            metrics.get("evaluation_seed"),
            metrics.get("projection_seed"),
        )
        for _, metrics in indexed.values()
    }
    if len(eval_shapes) != 1:
        raise PilotValidationError(
            "Evaluation data/projection configuration is not fixed across all runs: "
            f"{sorted(eval_shapes, key=str)}"
        )
    eval_shape = next(iter(eval_shapes))
    if eval_shape[1] is None:
        raise PilotValidationError(
            "config.fixed_eval_seed must be set for conditional fixed evaluation"
        )
    for index, value in enumerate(eval_shape):
        _finite_number(value, f"fixed evaluation field {index}")
    learning_rates = {
        _finite_number(metrics["config"].get("learning_rate"), f"{name}.config.learning_rate")
        for name, metrics in indexed.values()
    }
    if len(learning_rates) != 1:
        raise PilotValidationError(
            f"Training learning_rate is not fixed across all runs: {sorted(learning_rates)}"
        )
    return indexed


def _format_identity(identity: tuple[int, float, int]) -> str:
    return f"bs{identity[0]}/beta={identity[1]:g}/seed={identity[2]}"


def _aggregate(
    seed_runs: dict[int, dict[str, Any]],
    metric: str,
    baseline_runs: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    values = {str(seed): float(seed_runs[seed][metric]) for seed in sorted(seed_runs)}
    deltas = {
        str(seed): values[str(seed)] - float(baseline_runs[seed][metric])
        for seed in sorted(seed_runs)
    }
    delta_values = list(deltas.values())
    ci = exact_bootstrap_ci(delta_values)
    return {
        "mean": float(np.mean(list(values.values()))),
        "std": float(np.std(list(values.values()), ddof=1)),
        "values": list(values.values()),
        "values_by_seed": values,
        "paired_delta_mean": float(np.mean(delta_values)),
        "paired_deltas_by_seed": deltas,
        "paired_delta_ci_95": list(ci),
        "paired_delta_interval": {
            "confidence": 0.95,
            "method": "exact_percentile_bootstrap_all_n_to_n_resamples",
            "resamples": len(delta_values) ** len(delta_values),
            "lower": ci[0],
            "upper": ci[1],
        },
    }


def _simple_aggregate(seed_runs: dict[int, dict[str, Any]], metric: str) -> dict[str, Any]:
    values = {str(seed): float(seed_runs[seed][metric]) for seed in sorted(seed_runs)}
    return {
        "mean": float(np.mean(list(values.values()))),
        "std": float(np.std(list(values.values()), ddof=1)),
        "values": list(values.values()),
        "values_by_seed": values,
    }


def analyze_pilot_results(
    raw_results: dict[str, dict[str, Any]],
    betas: list[float] | None = None,
    seeds: list[int] | None = None,
    *,
    expected_steps: int = DEFAULT_STEPS,
    include_ablation: bool = True,
) -> dict[str, Any]:
    """Validate and aggregate the pilot using paired, seed-labeled comparisons."""
    betas = DEFAULT_BETAS if betas is None else betas
    seeds = DEFAULT_SEEDS if seeds is None else seeds
    indexed = validate_pilot_results(
        raw_results,
        betas,
        seeds,
        expected_steps=expected_steps,
        include_ablation=include_ablation,
    )
    conditions: dict[tuple[int, float], dict[int, dict[str, Any]]] = {}
    source_runs: dict[tuple[int, float], dict[int, str]] = {}
    for (batch_size, beta, seed), (run_name, metrics) in indexed.items():
        conditions.setdefault((batch_size, beta), {})[seed] = metrics
        source_runs.setdefault((batch_size, beta), {})[seed] = run_name

    summaries: dict[str, dict[str, Any]] = {}
    for batch_size, beta in sorted(conditions):
        seed_runs = conditions[(batch_size, beta)]
        baseline = conditions[(batch_size, 0.0)]
        name = f"bs{batch_size}_beta_{beta:g}".replace(".", "_")
        summary: dict[str, Any] = {
            "batch_size": batch_size,
            "beta": beta,
            "seeds": sorted(seed_runs),
            "source_runs_by_seed": {
                str(seed): source_runs[(batch_size, beta)][seed] for seed in sorted(seed_runs)
            },
            "empirical_w2": _aggregate(
                seed_runs, "empirical_wasserstein_distance", baseline
            ),
            "sliced_w2": _aggregate(seed_runs, "sliced_wasserstein_distance", baseline),
            "mean_eigenvalue_spread": _aggregate(
                seed_runs, "mean_eigenvalue_spread", baseline
            ),
            "max_eigenvalue_spread": _simple_aggregate(
                seed_runs, "max_eigenvalue_spread"
            ),
            "mode_entropy": _simple_aggregate(seed_runs, "mode_entropy"),
            "loss_cfm": _simple_aggregate(seed_runs, "final_loss_cfm"),
            "loss_iso": _simple_aggregate(seed_runs, "final_loss_iso"),
            "elapsed_seconds": _simple_aggregate(seed_runs, "elapsed_seconds"),
        }
        modes = {str(seed): int(seed_runs[seed]["modes_covered"]) for seed in sorted(seed_runs)}
        missing = {str(seed): int(seed_runs[seed]["missing_modes"]) for seed in sorted(seed_runs)}
        summary["modes_covered"] = {
            "mean": float(np.mean(list(modes.values()))),
            "min": min(modes.values()),
            "max": max(modes.values()),
            "values_by_seed": modes,
        }
        summary["missing_modes"] = {
            "mean": float(np.mean(list(missing.values()))),
            "total": sum(missing.values()),
            "values_by_seed": missing,
        }
        for output_key, metric in (
            ("pareto_sliced_w2", "pareto_step_sliced_wasserstein_distance"),
            ("pareto_empirical_w2", "pareto_step_empirical_wasserstein_distance"),
        ):
            statistics = {}
            for step in STEP_COUNTS:
                step_runs = {
                    seed: {metric: seed_runs[seed][metric][step]} for seed in seed_runs
                }
                step_baseline = {
                    seed: {metric: baseline[seed][metric][step]} for seed in baseline
                }
                statistics[step] = _aggregate(step_runs, metric, step_baseline)
            summary[output_key] = {
                step: step_statistics["mean"]
                for step, step_statistics in statistics.items()
            }
            summary[f"{output_key}_statistics"] = statistics
        summaries[name] = summary

    candidates = []
    for name, summary in summaries.items():
        if summary["batch_size"] != 256 or summary["beta"] == 0.0:
            continue
        endpoint = summary["empirical_w2"]
        low_steps = summary["pareto_empirical_w2_statistics"]
        reasons = []
        if summary["missing_modes"]["total"] != 0:
            reasons.append("one_or_more_missing_modes")
        if endpoint["paired_delta_mean"] > 0:
            reasons.append("endpoint_empirical_w2_sacrifice")
        sacrificed_steps = [
            step for step in LOW_STEP_COUNTS if low_steps[step]["paired_delta_mean"] > 0
        ]
        if sacrificed_steps:
            reasons.append("low_step_empirical_w2_sacrifice:" + ",".join(sacrificed_steps))
        if endpoint["paired_delta_interval"]["upper"] > 0:
            reasons.append("endpoint_ci_upper_above_zero")
        candidate = {
            "condition": name,
            "beta": summary["beta"],
            "eligible_for_promotion": not reasons,
            "promotion_failures": reasons,
            "delta_w2": endpoint["paired_delta_mean"],
            "delta_w2_by_seed": endpoint["paired_deltas_by_seed"],
            "delta_w2_ci": endpoint["paired_delta_ci_95"],
            "paired_endpoint_interval": endpoint["paired_delta_interval"],
            "delta_low_step_empirical_w2": {
                step: low_steps[step]["paired_delta_mean"] for step in LOW_STEP_COUNTS
            },
            "delta_low_step_empirical_w2_by_seed": {
                step: low_steps[step]["paired_deltas_by_seed"] for step in LOW_STEP_COUNTS
            },
            "delta_low_step_sw2": summary["pareto_sliced_w2_statistics"]["5"][
                "paired_delta_mean"
            ],
            "mean_eig_spread": summary["mean_eigenvalue_spread"]["mean"],
            "delta_eig_spread": summary["mean_eigenvalue_spread"]["paired_delta_mean"],
            "missing_modes": summary["missing_modes"]["total"],
        }
        candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            not item["eligible_for_promotion"],
            item["delta_w2"],
            item["delta_low_step_empirical_w2"]["5"],
            item["beta"],
        )
    )
    promoted = [candidate for candidate in candidates if candidate["eligible_for_promotion"]]

    raw_runs = {
        run_name: {
            "source_metrics": f"{run_name}/metrics.json",
            "identity": {
                "batch_size": identity[0],
                "beta": identity[1],
                "seed": identity[2],
            },
            "metrics": metrics,
        }
        for identity, (run_name, metrics) in sorted(indexed.items())
    }
    return {
        "schema": "quantumflow/beta-pilot-analysis/2",
        "design": {
            "betas": [float(beta) for beta in betas],
            "seeds": [int(seed) for seed in seeds],
            "steps": expected_steps,
            "batch_sizes": [128, 256] if include_ablation else [256],
            "ablation_betas": ABLATION_BETAS if include_ablation else [],
            "ode_step_counts": list(STEP_COUNTS),
            "low_step_counts": list(LOW_STEP_COUNTS),
            "fixed_evaluation": True,
        },
        "inference": {
            "unit": "paired_training_seed",
            "n_pairs": len(seeds),
            "interval": "exact percentile bootstrap over all n^n resamples",
            "interpretation": (
                "Exploratory small-n inference (three paired seeds), conditional on the fixed "
                "evaluation sample/projections and not adjusted for multiple comparisons."
            ),
        },
        "promotion_rule": {
            "preregistered": True,
            "requirements": [
                "zero missing modes in every seed",
                "paired mean endpoint empirical W2 delta <= 0",
                "paired mean empirical W2 delta <= 0 at each low-step count",
                "95% exact-resampling endpoint delta interval upper bound <= 0",
            ],
        },
        "summary_by_condition": summaries,
        "promoted_candidates": promoted,
        "all_candidates": candidates,
        "raw_runs": raw_runs,
    }


def collect_pilot_results(
    base_dir: Path,
    *,
    require_any: bool = True,
) -> dict[str, dict[str, Any]]:
    """Collect metrics by directory while deriving all identities from metadata."""
    results = {}
    for metrics_path in sorted(base_dir.glob("pilot_*/metrics.json")):
        try:
            results[metrics_path.parent.name] = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise PilotValidationError(f"Cannot read {metrics_path}: {exc}") from exc
    if require_any and not results:
        raise FileNotFoundError(f"No pilot metrics found under {base_dir}")
    return results


def save_analysis(
    base_dir: Path,
    raw_results: dict[str, dict[str, Any]],
    analysis: dict[str, Any],
) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "pilot_raw_results.json").write_text(
        json.dumps(raw_results, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (base_dir / "pilot_analysis.json").write_text(
        json.dumps(analysis, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def run_pilot(
    betas: list[float] | None = None,
    seeds: list[int] | None = None,
    steps: int = DEFAULT_STEPS,
    base_output_dir: Path | None = None,
    include_ablation: bool = True,
) -> dict[str, Any]:
    betas = DEFAULT_BETAS if betas is None else betas
    seeds = DEFAULT_SEEDS if seeds is None else seeds
    base_dir = base_output_dir or Path("outputs/transport/pilot")
    base_dir.mkdir(parents=True, exist_ok=True)
    runs = [(beta, seed, 256) for beta in betas for seed in seeds]
    if include_ablation:
        runs.extend((beta, seed, 128) for beta in ABLATION_BETAS for seed in seeds)

    raw_results = {}
    for index, (beta, seed, batch_size) in enumerate(runs, 1):
        token = f"{beta:g}".replace(".", "_")
        run_name = f"pilot_beta_{token}_s{seed}_bs{batch_size}"
        print(f"[{index}/{len(runs)}] {run_name}")
        raw_results[run_name] = train_and_eval(
            beta=beta,
            num_steps=steps,
            batch_size=batch_size,
            seed=seed,
            output_dir=base_dir / run_name,
        )
    analysis = analyze_pilot_results(
        raw_results,
        betas,
        seeds,
        expected_steps=steps,
        include_ablation=include_ablation,
    )
    save_analysis(base_dir, raw_results, analysis)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/transport/pilot"))
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Validate and aggregate existing metrics without training or gallery export",
    )
    args = parser.parse_args()
    if args.analyze_only:
        raw = collect_pilot_results(args.output_dir)
        analysis = analyze_pilot_results(
            raw,
            expected_steps=args.steps,
            include_ablation=not args.no_ablation,
        )
        save_analysis(args.output_dir, raw, analysis)
        return
    run_pilot(
        steps=args.steps,
        base_output_dir=args.output_dir,
        include_ablation=not args.no_ablation,
    )


if __name__ == "__main__":
    main()
