"""Experiment runner for Isotropic-Hessian OT-CFM."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from quantumflow import expdash
from quantumflow.ot_cfm import (
    cfm_loss_step,
    compute_eigenvalue_spread,
    compute_mode_metrics,
    empirical_wasserstein_distance,
    exact_ot_coupling,
    init_potential_network,
    integrate_ode,
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
) -> dict[str, float | list[float] | dict[str, float]]:
    if num_steps < 1 or batch_size < 1 or eval_samples < 1:
        raise ValueError("num_steps, batch_size, and eval_samples must be positive")
    if beta < 0:
        raise ValueError("beta must be non-negative")

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

    print(f"Starting training with beta={beta}, num_steps={num_steps}, batch_size={batch_size}")
    start_time = time.time()
    for step in range(1, num_steps + 1):
        key, k_t, k_x0, k_x1 = jax.random.split(key, 4)
        t = jax.random.uniform(k_t, (batch_size,))
        x0_raw = numpy_rng.standard_normal((batch_size, 2))
        x1_raw = np.asarray(sample_8gaussians(k_x1, batch_size))
        x0_ot, x1_ot = exact_ot_coupling(x0_raw, x1_raw)

        params, opt_state, aux = update_step(
            params,
            opt_state,
            t,
            jnp.array(x0_ot, dtype=jnp.float32),
            jnp.array(x1_ot, dtype=jnp.float32),
        )

        if step % 50 == 0 or step == num_steps:
            loss = float(aux["loss"])
            loss_cfm = float(aux["loss_cfm"])
            loss_iso = float(aux["loss_iso"])
            expdash.report(
                step=step,
                total=num_steps,
                loss=loss,
                loss_cfm=loss_cfm,
                loss_iso=loss_iso,
                penalty=loss_iso,
            )
            print(
                f"[{step}/{num_steps}] loss: {loss:.5f} "
                f"(cfm: {loss_cfm:.5f}, iso: {loss_iso:.5f})"
            )

    training_seconds = time.time() - start_time
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
        },
        "beta": beta,
        "num_steps": num_steps,
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
        "pareto_step_sliced_wasserstein_distance": pareto_sliced,
        "pareto_step_empirical_wasserstein_distance": pareto_w2,
        "ode_function_evaluations": ode_function_evaluations,
        "evaluation_seed": eval_seed,
        "projection_seed": projection_seed,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "elapsed_seconds": training_seconds + evaluation_seconds,
    }

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        save_model_params(params, output_path / "model.npz")
        print(f"Saved metrics to {output_path / 'metrics.json'}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2D OT-CFM with Isotropic-Hessian penalty.")
    parser.add_argument("--beta", type=float, default=0.0, help="Penalty weight beta")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Experiment seed")
    args = parser.parse_args()

    train_and_eval(
        beta=args.beta,
        num_steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
