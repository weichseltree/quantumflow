"""Multi-seed isotropic-Hessian OT-CFM beta pilot sweep and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from run_transport import train_and_eval

DEFAULT_BETAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
DEFAULT_SEEDS = [0, 1, 2]
ABLATION_BETAS = [0.0, 1e-3, 1e-2]


def bootstrap_ci(
    deltas: list[float] | np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval for mean delta."""
    deltas_np = np.asarray(deltas, dtype=float)
    if len(deltas_np) == 0:
        return 0.0, 0.0
    if len(deltas_np) == 1:
        return float(deltas_np[0]), float(deltas_np[0])

    rng = np.random.default_rng(seed)
    boot_means = np.zeros(n_boot)
    n = len(deltas_np)
    for i in range(n_boot):
        sample = rng.choice(deltas_np, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    lower_p = (1.0 - ci) / 2.0 * 100.0
    upper_p = (1.0 + ci) / 2.0 * 100.0
    return float(np.percentile(boot_means, lower_p)), float(np.percentile(boot_means, upper_p))


def run_pilot(
    betas: list[float] | None = None,
    seeds: list[int] | None = None,
    steps: int = 2000,
    base_output_dir: Path | None = None,
    include_ablation: bool = True,
) -> dict:
    if betas is None:
        betas = DEFAULT_BETAS
    if seeds is None:
        seeds = DEFAULT_SEEDS

    base_dir = base_output_dir or Path("outputs/transport/pilot")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Define run matrix
    runs = []
    # Main grid at batch size 256
    for b in betas:
        for s in seeds:
            runs.append({"beta": b, "seed": s, "batch_size": 256})

    # Ablation grid at batch size 128
    if include_ablation:
        for b in ABLATION_BETAS:
            for s in seeds:
                runs.append({"beta": b, "seed": s, "batch_size": 128})

    print("=" * 80)
    print("STARTING MULTI-SEED ISOTROPIC-HESSIAN OT-CFM BETA PILOT")
    print(f"Total runs: {len(runs)}")
    print(f"Betas (BS=256): {betas}")
    print(f"Ablation Betas (BS=128): {ABLATION_BETAS if include_ablation else []}")
    print(f"Seeds: {seeds}")
    print(f"Steps per run: {steps}")
    print(f"Output root: {base_dir}")
    print("=" * 80)

    raw_results = {}

    for idx, r_cfg in enumerate(runs, 1):
        b = r_cfg["beta"]
        s = r_cfg["seed"]
        bs = r_cfg["batch_size"]
        b_str = f"beta_{b:g}".replace(".", "_")
        run_name = f"pilot_{b_str}_s{s}_bs{bs}"
        run_dir = base_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{idx}/{len(runs)}] Running {run_name} (beta={b}, seed={s}, bs={bs})...")
        res = train_and_eval(
            beta=b,
            num_steps=steps,
            batch_size=bs,
            seed=s,
            output_dir=run_dir,
        )
        raw_results[run_name] = res

    # Analyze results
    analysis = analyze_pilot_results(raw_results, betas, seeds)

    # Save raw and analyzed JSON summaries
    with open(base_dir / "pilot_raw_results.json", "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2)

    with open(base_dir / "pilot_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print(f"\nPilot results saved to: {base_dir / 'pilot_analysis.json'}")
    return analysis


def analyze_pilot_results(
    raw_results: dict[str, dict],
    betas: list[float],
    seeds: list[int],
) -> dict:
    """Compute aggregate statistics, paired seed-wise deltas, and Pareto ODE comparisons."""
    conditions = {}

    # Separate by batch_size and beta
    for run_name, r in raw_results.items():
        b = r["beta"]
        bs = r["config"]["batch_size"]
        s = r["seed"]
        key = (bs, b)
        if key not in conditions:
            conditions[key] = {}
        conditions[key][s] = r

    summary_by_condition = {}

    for (bs, b), seed_runs in conditions.items():
        cond_name = f"bs{bs}_beta_{b:g}".replace(".", "_")
        
        w2_vals = [seed_runs[s]["empirical_wasserstein_distance"] for s in seed_runs]
        sw2_vals = [seed_runs[s]["sliced_wasserstein_distance"] for s in seed_runs]
        eig_mean_vals = [seed_runs[s]["mean_eigenvalue_spread"] for s in seed_runs]
        eig_max_vals = [seed_runs[s]["max_eigenvalue_spread"] for s in seed_runs]
        modes_vals = [seed_runs[s]["modes_covered"] for s in seed_runs]
        missing_vals = [seed_runs[s]["missing_modes"] for s in seed_runs]
        entropy_vals = [seed_runs[s]["mode_entropy"] for s in seed_runs]
        loss_cfm_vals = [seed_runs[s]["final_loss_cfm"] for s in seed_runs]
        loss_iso_vals = [seed_runs[s]["final_loss_iso"] for s in seed_runs]
        time_vals = [seed_runs[s]["elapsed_seconds"] for s in seed_runs]

        # Paired deltas against beta=0 baseline with same batch size
        baseline_runs = conditions.get((bs, 0.0), {})
        paired_w2_deltas = []
        paired_sw2_deltas = []
        paired_eig_mean_deltas = []
        paired_eig_max_deltas = []

        for s in seed_runs:
            if s in baseline_runs:
                paired_w2_deltas.append(
                    seed_runs[s]["empirical_wasserstein_distance"]
                    - baseline_runs[s]["empirical_wasserstein_distance"]
                )
                paired_sw2_deltas.append(
                    seed_runs[s]["sliced_wasserstein_distance"]
                    - baseline_runs[s]["sliced_wasserstein_distance"]
                )
                paired_eig_mean_deltas.append(
                    seed_runs[s]["mean_eigenvalue_spread"]
                    - baseline_runs[s]["mean_eigenvalue_spread"]
                )
                paired_eig_max_deltas.append(
                    seed_runs[s]["max_eigenvalue_spread"]
                    - baseline_runs[s]["max_eigenvalue_spread"]
                )

        w2_ci = bootstrap_ci(paired_w2_deltas)
        sw2_ci = bootstrap_ci(paired_sw2_deltas)
        eig_mean_ci = bootstrap_ci(paired_eig_mean_deltas)

        # Multi-step Pareto averages
        step_counts = ["5", "10", "20", "50", "100"]
        pareto_sw2_mean = {}
        pareto_w2_mean = {}
        for sc in step_counts:
            pareto_sw2_mean[sc] = float(np.mean([
                seed_runs[s]["pareto_step_sliced_wasserstein_distance"].get(sc, 0.0)
                for s in seed_runs
            ]))
            pareto_w2_mean[sc] = float(np.mean([
                seed_runs[s]["pareto_step_empirical_wasserstein_distance"].get(sc, 0.0)
                for s in seed_runs
            ]))

        summary_by_condition[cond_name] = {
            "batch_size": bs,
            "beta": b,
            "seeds": list(seed_runs.keys()),
            "empirical_w2": {
                "mean": float(np.mean(w2_vals)),
                "std": float(np.std(w2_vals, ddof=1)) if len(w2_vals) > 1 else 0.0,
                "values": [float(v) for v in w2_vals],
                "paired_delta_mean": float(np.mean(paired_w2_deltas)) if paired_w2_deltas else 0.0,
                "paired_delta_ci_95": w2_ci,
            },
            "sliced_w2": {
                "mean": float(np.mean(sw2_vals)),
                "std": float(np.std(sw2_vals, ddof=1)) if len(sw2_vals) > 1 else 0.0,
                "values": [float(v) for v in sw2_vals],
                "paired_delta_mean": (
                    float(np.mean(paired_sw2_deltas)) if paired_sw2_deltas else 0.0
                ),
                "paired_delta_ci_95": sw2_ci,
            },
            "mean_eigenvalue_spread": {
                "mean": float(np.mean(eig_mean_vals)),
                "std": float(np.std(eig_mean_vals, ddof=1)) if len(eig_mean_vals) > 1 else 0.0,
                "paired_delta_mean": (
                    float(np.mean(paired_eig_mean_deltas))
                    if paired_eig_mean_deltas
                    else 0.0
                ),
                "paired_delta_ci_95": eig_mean_ci,
            },
            "max_eigenvalue_spread": {
                "mean": float(np.mean(eig_max_vals)),
                "std": float(np.std(eig_max_vals, ddof=1)) if len(eig_max_vals) > 1 else 0.0,
            },
            "modes_covered": {
                "mean": float(np.mean(modes_vals)),
                "min": int(np.min(modes_vals)),
                "max": int(np.max(modes_vals)),
            },
            "missing_modes": {
                "mean": float(np.mean(missing_vals)),
                "total": int(np.sum(missing_vals)),
            },
            "mode_entropy": {
                "mean": float(np.mean(entropy_vals)),
                "std": float(np.std(entropy_vals, ddof=1)) if len(entropy_vals) > 1 else 0.0,
            },
            "loss_cfm": {
                "mean": float(np.mean(loss_cfm_vals)),
                "std": float(np.std(loss_cfm_vals, ddof=1)) if len(loss_cfm_vals) > 1 else 0.0,
            },
            "loss_iso": {
                "mean": float(np.mean(loss_iso_vals)),
                "std": float(np.std(loss_iso_vals, ddof=1)) if len(loss_iso_vals) > 1 else 0.0,
            },
            "pareto_sliced_w2": pareto_sw2_mean,
            "pareto_empirical_w2": pareto_w2_mean,
            "elapsed_seconds": {
                "mean": float(np.mean(time_vals)),
                "std": float(np.std(time_vals, ddof=1)) if len(time_vals) > 1 else 0.0,
            },
        }

    # Identify candidate promotion based on empirical W2, low-step W2, and eigenvalue spread
    candidates = []
    for cond_name, s in summary_by_condition.items():
        if s["batch_size"] == 256 and s["beta"] > 0.0:
            delta_w2 = s["empirical_w2"]["paired_delta_mean"]
            delta_sw2_5s = (
                s["pareto_sliced_w2"]["5"]
                - summary_by_condition["bs256_beta_0"]["pareto_sliced_w2"]["5"]
            )
            candidates.append({
                "condition": cond_name,
                "beta": s["beta"],
                "delta_w2": delta_w2,
                "delta_w2_ci": s["empirical_w2"]["paired_delta_ci_95"],
                "delta_low_step_sw2": delta_sw2_5s,
                "mean_eig_spread": s["mean_eigenvalue_spread"]["mean"],
                "delta_eig_spread": s["mean_eigenvalue_spread"]["paired_delta_mean"],
                "missing_modes": s["missing_modes"]["total"],
            })

    # Sort candidates by combined improvement in W2 and low-step W2
    candidates.sort(key=lambda c: (c["missing_modes"], c["delta_w2"] + c["delta_low_step_sw2"]))

    return {
        "summary_by_condition": summary_by_condition,
        "promoted_candidates": candidates[:3],
        "all_candidates": candidates,
    }


def collect_pilot_results(base_dir: Path) -> dict[str, dict]:
    """Collect completed run metrics from a pilot output tree."""
    results = {}
    for metrics_path in sorted(base_dir.glob("pilot_*/metrics.json")):
        with metrics_path.open(encoding="utf-8") as handle:
            results[metrics_path.parent.name] = json.load(handle)
    if not results:
        raise FileNotFoundError(f"No pilot metrics found under {base_dir}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed isotropic-Hessian OT-CFM beta pilot.")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps per run")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/transport/pilot"),
        help="Base output directory",
    )
    parser.add_argument(
        "--no-ablation",
        action="store_true",
        help="Skip BS=128 ablation runs",
    )
    args = parser.parse_args()

    run_pilot(
        steps=args.steps,
        base_output_dir=args.output_dir,
        include_ablation=not args.no_ablation,
    )


if __name__ == "__main__":
    main()
