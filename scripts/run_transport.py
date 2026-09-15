"""Experiment runner for Isotropic-Hessian OT-CFM."""

from __future__ import annotations

import argparse
import json
import os
import platform
import hashlib
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from quantumflow import expdash
from quantumflow.checkpoint import load_checkpoint, save_checkpoint
from quantumflow.ot_cfm import (
    cfm_loss_step,
    compute_eigenvalue_spread,
    compute_mode_metrics,
    empirical_wasserstein_distance,
    exact_ot_coupling,
    init_potential_network,
    integrate_ode,
    load_model_params,
    sample_8gaussians,
    save_model_params,
    sliced_wasserstein_distance,
)


def train_and_eval(
    beta: float = 0.0,
    num_steps: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    output_dir: Path | None = None,
    eval_samples: int = 2048,
    fixed_eval_seed: int | None = 1_000_003,
    init_model: Path | None = None,
    step_offset: int | None = None,
    sample_offset: int | None = None,
) -> dict:
    if num_steps < 1 or batch_size < 1 or eval_samples < 1:
        raise ValueError("num_steps, batch_size, and eval_samples must be positive")
    if not np.isfinite(beta) or beta < 0 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("beta must be finite and non-negative; lr must be finite and positive")
    source_metrics = None
    if init_model is not None:
        source_metrics = json.loads(
            (init_model.parent / "metrics.json").read_text(encoding="utf-8")
        )
        for field, expected in (("seed", seed), ("beta", beta), ("batch_size", batch_size)):
            if source_metrics[field] != expected:
                raise ValueError(f"Exact continuation requires matching {field}")
        if source_metrics["config"]["learning_rate"] != lr:
            raise ValueError("Exact continuation requires matching learning rate")
        source_steps = source_metrics["end_step"]
        source_samples = source_metrics["end_samples"]
        if step_offset is not None and step_offset != source_steps:
            raise ValueError("step_offset must match parent end_step")
        if sample_offset is not None and sample_offset != source_samples:
            raise ValueError("sample_offset must match parent end_samples")
        step_offset, sample_offset = source_steps, source_samples
    elif step_offset or sample_offset:
        raise ValueError("Nonzero offsets require a parent checkpoint")
    step_offset = step_offset or 0
    if step_offset < 0:
        raise ValueError("step_offset must be non-negative")
    if sample_offset is None:
        sample_offset = step_offset * batch_size
    if sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")

    key = jax.random.key(seed)
    numpy_rng = np.random.default_rng(seed)
    key, subkey = jax.random.split(key)
    params = init_potential_network(subkey, in_dim=2, hidden_dims=(128, 128))
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def update_step(p, opt_s, t_batch, x0_batch, x1_batch):
        grad_fn = jax.value_and_grad(
            lambda p_: cfm_loss_step(p_, t_batch, x0_batch, x1_batch, beta=beta),
            has_aux=True,
        )
        (loss_val, aux), grads = grad_fn(p)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt_s, aux

    def advance(p, opt_s, rng_key):
        next_key, k_t, _, k_x1 = jax.random.split(rng_key, 4)
        t = jax.random.uniform(k_t, (batch_size,))
        x0_raw = numpy_rng.standard_normal((batch_size, 2))
        x1_raw = np.asarray(sample_8gaussians(k_x1, batch_size))
        x0_ot, x1_ot = exact_ot_coupling(x0_raw, x1_raw)
        new_p, new_opt_s, aux = update_step(
            p, opt_s, t, jnp.array(x0_ot, dtype=jnp.float32),
            jnp.array(x1_ot, dtype=jnp.float32),
        )
        return new_p, new_opt_s, next_key, aux

    replay_seconds = 0.0
    history = []
    if init_model is not None:
        checkpoint = init_model.parent / "checkpoint.npz"
        if checkpoint.is_file():
            (params, opt_state, key_data), meta = load_checkpoint(
                checkpoint, (params, opt_state, jax.random.key_data(key))
            )
            if meta["end_step"] != step_offset or meta["end_samples"] != sample_offset:
                raise ValueError("Checkpoint counters disagree with parent metrics")
            key = jax.random.wrap_key_data(key_data)
            numpy_rng.bit_generator.state = meta["numpy_rng"]
        else:
            # Older pilot runs only saved weights. Recover Adam and RNG state by
            # exact deterministic replay, inside this job's GPU scheduler lock.
            replay_start = time.perf_counter()
            for _ in range(step_offset):
                params, opt_state, key, _ = advance(params, opt_state, key)
            saved = load_model_params(init_model)
            for recovered, original in zip(
                jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(saved), strict=True
            ):
                np.testing.assert_allclose(recovered, original, atol=1e-6, rtol=1e-6)
            replay_seconds = time.perf_counter() - replay_start
            print(f"Recovered parent optimizer/RNG by verified replay in {replay_seconds:.2f}s")
        parent_history = init_model.parent / "progress.json"
        if parent_history.is_file():
            expdash.resume_history(parent_history)
            history = json.loads(parent_history.read_text(encoding="utf-8"))["history"]
        else:
            raise FileNotFoundError(f"Parent chart history missing: {parent_history}")

    print(f"Starting training with beta={beta}, num_steps={num_steps}, batch_size={batch_size}")
    start_time = time.time()
    for step in range(1, num_steps + 1):
        params, opt_state, key, aux = advance(params, opt_state, key)

        if step % 50 == 0 or step == num_steps:
            loss = float(aux["loss"])
            loss_cfm = float(aux["loss_cfm"])
            loss_iso = float(aux["loss_iso"])
            absolute_optimizer_step = step_offset + step
            absolute_samples = sample_offset + step * batch_size
            expdash.report(
                step=absolute_samples,
                total=sample_offset + num_steps * batch_size,
                step_offset=0,
                loss=loss,
                loss_cfm=loss_cfm,
                loss_iso=loss_iso,
                penalty=loss_iso,
                optimizer_step=absolute_optimizer_step,
                samples_seen=absolute_samples,
                progress_unit="samples",
            )
            values = {
                "loss": loss, "loss_cfm": loss_cfm, "loss_iso": loss_iso,
                "optimizer_step": absolute_optimizer_step, "samples_seen": absolute_samples,
            }
            history.append([time.time(), absolute_samples, values])
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                progress = {
                    "history": history, "step": absolute_samples,
                    "total": sample_offset + num_steps * batch_size,
                    "values": {**values, "progress_unit": "samples"},
                }
                temporary = output_dir / "progress.tmp"
                temporary.write_text(json.dumps(progress, allow_nan=False), encoding="utf-8")
                temporary.replace(output_dir / "progress.json")
            print(
                f"[{step}/{num_steps}] loss: {loss:.5f} "
                f"(cfm: {loss_cfm:.5f}, iso: {loss_iso:.5f})"
            )

    training_seconds = time.time() - start_time
    if output_dir:
        save_model_params(params, output_dir / "model.npz")
        save_checkpoint(
            output_dir / "checkpoint.npz",
            (params, opt_state, jax.random.key_data(key)),
            {
                "end_step": step_offset + num_steps,
                "end_samples": sample_offset + num_steps * batch_size,
                "numpy_rng": numpy_rng.bit_generator.state,
            },
        )
    print(f"Training completed in {training_seconds:.2f}s. Evaluating...")
    evaluation_start = time.time()

    # Evaluation
    # 1. Sample quality: integrate x0 -> x1 with RK4 (50 steps)
    if fixed_eval_seed is not None:
        eval_seed = fixed_eval_seed
        projection_seed = fixed_eval_seed + 1_000_000
    else:
        eval_seed = seed + 1_000_003
        projection_seed = seed + 2_000_003

    k_eval_x0 = jax.random.key(eval_seed)
    k_eval_target = jax.random.key(eval_seed + 1)
    k_projection = jax.random.key(projection_seed)
    x0_eval = jax.random.normal(k_eval_x0, shape=(eval_samples, 2))
    target_samples = sample_8gaussians(k_eval_target, eval_samples)

    generated_samples, trajectory = integrate_ode(params, x0_eval, num_steps=50)
    sliced_distance = float(
        sliced_wasserstein_distance(generated_samples, target_samples, key=k_projection)
    )
    wasserstein_distance = empirical_wasserstein_distance(generated_samples, target_samples)
    mode_metrics = compute_mode_metrics(generated_samples)

    # 2. Eigenvalue spread along trajectory
    mean_spreads = []
    max_spreads = []
    for t_idx, x_t in enumerate(trajectory):
        t_val = t_idx / (len(trajectory) - 1)
        spread_dict = compute_eigenvalue_spread(params, x_t[:128], t=t_val)
        mean_spreads.append(float(spread_dict["mean"]))
        max_spreads.append(float(spread_dict["max"]))
    mean_eig_spread = float(np.mean(mean_spreads))
    max_eig_spread = float(np.max(max_spreads))

    # 3. ODE step count vs quality (Pareto analysis)
    step_counts = [5, 10, 20, 50, 100]
    pareto_sliced = {}
    pareto_w2 = {}
    ode_function_evaluations = {}
    for sc in step_counts:
        gen_sc, _ = integrate_ode(params, x0_eval, num_steps=sc)
        swd_sc = float(
            sliced_wasserstein_distance(gen_sc, target_samples, key=k_projection)
        )
        w2_sc = float(empirical_wasserstein_distance(gen_sc, target_samples))
        pareto_sliced[str(sc)] = swd_sc
        pareto_w2[str(sc)] = w2_sc
        ode_function_evaluations[str(sc)] = 4 * sc
    evaluation_seconds = time.time() - evaluation_start

    metrics = {
        "schema_version": 2,
        "seed": seed,
        "config": {
            "beta": beta,
            "num_steps": num_steps,
            "batch_size": batch_size,
            "learning_rate": lr,
            "eval_samples": eval_samples,
            "ode_step_counts": step_counts,
            "fixed_eval_seed": fixed_eval_seed,
            "init_model": str(init_model) if init_model else None,
            "step_offset": step_offset,
            "sample_offset": sample_offset,
        },
        "beta": beta,
        "num_steps": num_steps,
        "start_step": step_offset,
        "end_step": step_offset + num_steps,
        "start_samples": sample_offset,
        "end_samples": sample_offset + num_steps * batch_size,
        "batch_size": batch_size,
        "final_loss": float(aux["loss"]),
        "final_loss_cfm": float(aux["loss_cfm"]),
        "final_loss_iso": float(aux["loss_iso"]),
        "sliced_wasserstein_distance": sliced_distance,
        "empirical_wasserstein_distance": wasserstein_distance,
        "mean_eigenvalue_spread": mean_eig_spread,
        "max_eigenvalue_spread": max_eig_spread,
        "trajectory_eigenvalue_spreads": mean_spreads,
        "trajectory_max_eigenvalue_spreads": max_spreads,
        "modes_covered": mode_metrics["modes_covered"],
        "missing_modes": mode_metrics["missing_modes"],
        "mode_counts": mode_metrics["mode_counts"],
        "mode_entropy": mode_metrics["mode_entropy"],
        "unassigned_samples": mode_metrics["unassigned_samples"],
        "pareto_step_sliced_wasserstein_distance": pareto_sliced,
        "pareto_step_empirical_wasserstein_distance": pareto_w2,
        "ode_function_evaluations": ode_function_evaluations,
        "evaluation_seed": eval_seed,
        "projection_seed": projection_seed,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "elapsed_seconds": training_seconds + evaluation_seconds,
        "replay_seconds": replay_seconds,
        "provenance": {
            "python": platform.python_version(), "jax": jax.__version__,
            "numpy": np.__version__, "optax": optax.__version__,
            "device": str(jax.devices()[0]),
            "exp_name": os.environ.get("EXP_NAME"),
            "exp_sweep": os.environ.get("EXP_SWEEP"),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "evaluation_source_sha256": hashlib.sha256(np.asarray(x0_eval).tobytes()).hexdigest(),
            "evaluation_target_sha256": hashlib.sha256(
                np.asarray(target_samples).tobytes()
            ).hexdigest(),
        },
    }

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "metrics.tmp", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, allow_nan=False)
        (output_path / "metrics.tmp").replace(output_path / "metrics.json")
        print(f"Saved metrics to {output_path / 'metrics.json'}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2D OT-CFM with Isotropic-Hessian penalty.")
    parser.add_argument("--beta", type=float, default=0.0, help="Penalty weight beta")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Experiment seed")
    parser.add_argument(
        "--init-model",
        type=Path,
        default=None,
        help="Portable model archive to continue from",
    )
    parser.add_argument(
        "--step-offset",
        type=int,
        default=None,
        help="Completed steps before this continuation",
    )
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=None,
        help="Completed samples before this continuation",
    )
    args = parser.parse_args()

    train_and_eval(
        beta=args.beta,
        num_steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
        init_model=args.init_model,
        step_offset=args.step_offset,
        sample_offset=args.sample_offset,
    )


if __name__ == "__main__":
    main()
