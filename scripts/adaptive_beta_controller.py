"""Reprioritize the pilot and extend promising runs by successive halving."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "outputs" / "transport" / "pilot"
ADAPTIVE_DIR = ROOT / "outputs" / "transport" / "adaptive"
SWEEP = "transport-beta-adaptive"
PYTHON = ROOT / ".venv" / "bin" / "python"
DEFAULT_SEEDS = [0, 1, 2]


def beta_token(beta: float) -> str:
    return f"{beta:g}".replace(".", "_")


def run_name(beta: float, seed: int, batch_size: int = 256) -> str:
    return f"pilot_beta_{beta_token(beta)}_s{seed}_bs{batch_size}"


def exp_name(beta: float, seed: int, batch_size: int = 256) -> str:
    return f"pilot-b{beta_token(beta)}-s{seed}-bs{batch_size}"


def load_metrics(directory: Path) -> dict | None:
    path = directory / "metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def completed_population(root: Path, betas: list[float]) -> dict[float, dict[int, dict]]:
    population = {}
    for beta in betas:
        seed_metrics = {}
        for seed in DEFAULT_SEEDS:
            metrics = load_metrics(root / run_name(beta, seed))
            if metrics is not None:
                seed_metrics[seed] = metrics
        if len(seed_metrics) == len(DEFAULT_SEEDS):
            population[beta] = seed_metrics
    return population


def score_population(seed_metrics: dict[int, dict]) -> float:
    """Lower is better: endpoint W2, low-step W2, anisotropy, then mode loss."""
    endpoint = np.mean(
        [metrics["empirical_wasserstein_distance"] for metrics in seed_metrics.values()]
    )
    low_step = np.mean(
        [
            metrics["pareto_step_empirical_wasserstein_distance"]["5"]
            for metrics in seed_metrics.values()
        ]
    )
    spread = np.mean(
        [metrics["mean_eigenvalue_spread"] for metrics in seed_metrics.values()]
    )
    missing = sum(metrics["missing_modes"] for metrics in seed_metrics.values())
    return float(endpoint + low_step + 0.25 * spread + 10.0 * missing)


def reprioritize_initial_grid() -> None:
    """Move informative middle betas ahead without cancelling any run."""
    priority = {
        1e-3: 10,
        3e-3: 10,
        1e-2: 10,
        3e-2: 5,
        1e-1: 5,
        3e-1: 0,
        1.0: 0,
    }
    for beta, value in priority.items():
        for seed in DEFAULT_SEEDS:
            subprocess.run(
                ["expprio", exp_name(beta, seed), str(value)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def launch_continuation(
    *,
    beta: float,
    seed: int,
    source_dir: Path,
    target_steps: int,
    source_steps: int,
    priority: int,
) -> None:
    stage_dir = ADAPTIVE_DIR / f"steps_{target_steps}" / run_name(beta, seed)
    stage_dir.mkdir(parents=True, exist_ok=True)
    if (stage_dir / "metrics.json").is_file():
        return

    name = f"adaptive-b{beta_token(beta)}-s{seed}-to{target_steps}"
    command = [
        "exp",
        "run",
        name,
        "--prio",
        str(priority),
        "--sweep",
        SWEEP,
        "--lane",
        "gpu",
        "--log",
        str(stage_dir / "train.log"),
        "--",
        "env",
        "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        str(PYTHON),
        "scripts/run_transport.py",
        "--beta",
        str(beta),
        "--seed",
        str(seed),
        "--batch-size",
        "256",
        "--steps",
        str(target_steps - source_steps),
        "--step-offset",
        str(source_steps),
        "--sample-offset",
        str(source_steps * 256),
        "--init-model",
        str(source_dir / "model.npz"),
        "--output-dir",
        str(stage_dir),
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode not in (0, 2):
        raise RuntimeError(f"Failed to launch {name}: exit {result.returncode}")


def wait_for_population(root: Path, betas: list[float], poll_seconds: int = 15):
    while True:
        population = completed_population(root, betas)
        if len(population) == len(betas):
            return population
        time.sleep(poll_seconds)


def main() -> None:
    ADAPTIVE_DIR.mkdir(parents=True, exist_ok=True)
    reprioritize_initial_grid()

    # Preserve the fixed pilot, but evaluate the informative center first.
    exploratory_betas = [0.0, 1e-3, 3e-3, 1e-2, 3e-2]
    stage_2000 = wait_for_population(PILOT_DIR, exploratory_betas)
    ranked = sorted(
        (beta for beta in exploratory_betas if beta > 0.0),
        key=lambda beta: score_population(stage_2000[beta]),
    )
    promoted_4000 = [0.0, *ranked[:2]]

    for beta in promoted_4000:
        for seed in DEFAULT_SEEDS:
            launch_continuation(
                beta=beta,
                seed=seed,
                source_dir=PILOT_DIR / run_name(beta, seed),
                target_steps=4000,
                source_steps=2000,
                priority=5,
            )

    stage_4000_root = ADAPTIVE_DIR / "steps_4000"
    stage_4000 = wait_for_population(stage_4000_root, promoted_4000)
    leader = min(
        (beta for beta in promoted_4000 if beta > 0.0),
        key=lambda beta: score_population(stage_4000[beta]),
    )

    for beta in [0.0, leader]:
        for seed in DEFAULT_SEEDS:
            launch_continuation(
                beta=beta,
                seed=seed,
                source_dir=stage_4000_root / run_name(beta, seed),
                target_steps=8000,
                source_steps=4000,
                priority=5,
            )

    summary = {
        "schema": "quantumflow/adaptive-beta/1",
        "ranking_at_2000": [
            {"beta": beta, "score": score_population(stage_2000[beta])}
            for beta in ranked
        ],
        "promoted_to_4000": promoted_4000,
        "promoted_to_8000": [0.0, leader],
        "progress_unit": "samples",
    }
    (ADAPTIVE_DIR / "adaptive_plan.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
