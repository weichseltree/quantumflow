"""Reprioritize the pilot and extend promising runs by successive halving."""

from __future__ import annotations

import json
import shlex
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
    if beta == 1:
        return "1_0"
    return f"{beta:g}".replace(".", "_")


def run_name(beta: float, seed: int, batch_size: int = 256) -> str:
    return f"pilot_beta_{beta_token(beta)}_s{seed}_bs{batch_size}"


def exp_name(beta: float, seed: int, batch_size: int = 256) -> str:
    return f"pilot-b{beta_token(beta)}-s{seed}-bs{batch_size}"


def load_metrics(directory: Path) -> dict | None:
    path = directory / "metrics.json"
    if not path.is_file() or not (directory / "model.npz").is_file():
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
    score = float(endpoint + low_step + 0.25 * spread + 10.0 * missing)
    if not np.isfinite(score):
        raise ValueError("Cannot rank a non-finite population")
    return score


def find_record(name: str) -> tuple[Path, dict] | None:
    records = []
    for path in (Path.home() / ".exp_status").glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        if record.get("name") == name and "status" in record:
            records.append((path, record))
    return max(records, key=lambda item: item[0].name) if records else None


def preserve_parent_history(source_dir: Path, name: str) -> None:
    if (source_dir / "progress.json").is_file():
        return
    record = find_record(name)
    if record is None:
        raise RuntimeError(f"No ExpDash lineage found for {name}")
    history_path = record[0].with_suffix(".metrics")
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    metrics = load_metrics(source_dir)
    if (payload.get("values", {}).get("progress_unit") != "samples"
            or metrics is None or payload.get("step") != metrics["end_samples"]):
        raise ValueError(f"Parent sample history is incomplete for {name}")
    (source_dir / "progress.json").write_text(json.dumps(payload), encoding="utf-8")


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
                check=True,
                stdout=subprocess.DEVNULL,
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
    if load_metrics(stage_dir) is not None:
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
    parent_name = (exp_name(beta, seed) if source_steps == 2000
                   else f"adaptive-b{beta_token(beta)}-s{seed}-to{source_steps}")
    preserve_parent_history(source_dir, parent_name)
    existing = find_record(name)
    expected_cmd = command[command.index("--") + 1:]
    if existing is not None and existing[1]["status"] in ("running", "queued", "success"):
        if shlex.split(existing[1]["cmd"]) != expected_cmd:
            raise RuntimeError(f"Existing {name} has a different command")
        return
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        record = find_record(name)
        if (result.returncode != 2 or record is None
                or record[1]["status"] not in ("running", "queued", "success")
                or shlex.split(record[1]["cmd"]) != expected_cmd):
            raise RuntimeError(f"Failed to launch {name}: exit {result.returncode}")


def wait_for_population(
    root: Path, betas: list[float], poll_seconds: int = 15, timeout_seconds: int = 14400
):
    deadline = time.monotonic() + timeout_seconds
    while True:
        population = completed_population(root, betas)
        if len(population) == len(betas):
            return population
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Population incomplete at {root}: {sorted(set(betas) - population.keys())}")
        for beta in betas:
            for seed in DEFAULT_SEEDS:
                if load_metrics(root / run_name(beta, seed)) is not None:
                    continue
                name = (exp_name(beta, seed) if root == PILOT_DIR else
                        f"adaptive-b{beta_token(beta)}-s{seed}-to{root.name.split('_')[-1]}")
                record = find_record(name)
                if record is not None and record[1]["status"] in (
                    "failed", "error", "cancelled", "canceled", "success"
                ):
                    raise RuntimeError(f"{name} ended {record[1]['status']} without complete artifacts")
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
    summary = {
        "schema": "quantumflow/adaptive-beta/2",
        "method": "successive halving with exact optimizer/RNG continuation, not PBT",
        "ranking_at_2000": [
            {"beta": beta, "score": score_population(stage_2000[beta])} for beta in ranked
        ],
        "promoted_to_4000": promoted_4000,
        "progress_unit": "samples",
        "status": "extending_to_4000",
    }
    save_plan(summary)

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
    summary.update(
        promoted_to_8000=[0.0, leader],
        ranking_at_4000=[
            {"beta": beta, "score": score_population(stage_4000[beta])}
            for beta in promoted_4000
        ],
        status="extending_to_8000",
    )
    save_plan(summary)

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

    final = wait_for_population(ADAPTIVE_DIR / "steps_8000", [0.0, leader])
    summary.update(
        status="complete",
        scores_at_8000=[
            {"beta": beta, "score": score_population(final[beta])} for beta in [0.0, leader]
        ],
    )
    save_plan(summary)


def save_plan(summary: dict) -> None:
    temporary = ADAPTIVE_DIR / "adaptive_plan.tmp"
    temporary.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    temporary.replace(ADAPTIVE_DIR / "adaptive_plan.json")


if __name__ == "__main__":
    main()
