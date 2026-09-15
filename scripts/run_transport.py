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
    exact_ot_coupling,
    init_potential_network,
    integrate_ode,
    sample_8gaussians,
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
) -> dict[str, float | list[float] | dict[str, float]]:
    key = jax.random.key(seed)
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
        x0_raw = np.random.randn(batch_size, 2)
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
                beta=beta,
            )
            print(f"[{step}/{num_steps}] loss: {loss:.5f} (cfm: {loss_cfm:.5f}, iso: {loss_iso:.5f})")

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f}s. Evaluating...")

    # Evaluation
    # 1. Sample quality: integrate x0 -> x1 with RK4 (50 steps)
    key, k_eval_x0, k_eval_target, k_swd = jax.random.split(key, 4)
    x0_eval = jax.random.normal(k_eval_x0, shape=(eval_samples, 2))
    target_samples = sample_8gaussians(k_eval_target, eval_samples)

    generated_samples, trajectory = integrate_ode(params, x0_eval, num_steps=50)
    w2_dist = float(sliced_wasserstein_distance(generated_samples, target_samples, key=k_swd))

    # 2. Eigenvalue spread along trajectory
    spreads = []
    for t_idx, x_t in enumerate(trajectory):
        t_val = t_idx / (len(trajectory) - 1)
        spread_val = compute_eigenvalue_spread(params, x_t[:128], t=t_val)
        spreads.append(float(spread_val))
    mean_eig_spread = float(np.mean(spreads))

    # 3. ODE step count vs quality (Pareto analysis)
    step_counts = [5, 10, 20, 50, 100]
    pareto_w2 = {}
    for sc in step_counts:
        key, k_swd_sc = jax.random.split(key)
        gen_sc, _ = integrate_ode(params, x0_eval, num_steps=sc)
        swd_sc = float(sliced_wasserstein_distance(gen_sc, target_samples, key=k_swd_sc))
        pareto_w2[str(sc)] = swd_sc

    metrics = {
        "beta": beta,
        "num_steps": num_steps,
        "final_loss": float(aux["loss"]),
        "final_loss_cfm": float(aux["loss_cfm"]),
        "final_loss_iso": float(aux["loss_iso"]),
        "w2_distance": w2_dist,
        "mean_eigenvalue_spread": mean_eig_spread,
        "trajectory_eigenvalue_spreads": spreads,
        "pareto_step_w2": pareto_w2,
        "elapsed_seconds": elapsed,
    }

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {output_path / 'metrics.json'}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2D OT-CFM with Isotropic-Hessian penalty.")
    parser.add_argument("--beta", type=float, default=0.0, help="Penalty weight beta")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    train_and_eval(
        beta=args.beta,
        num_steps=args.steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
