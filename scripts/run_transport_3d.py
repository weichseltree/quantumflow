"""3D Optimal Transport Conditional Flow Matching (OT-CFM) Experiment Runner.

Supports 3D potential flow learning v(t, r) = grad_r Phi(t, r) with 3D
Isotropic-Hessian regularization and ExpDash progress tracking.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from quantumflow import expdash
from quantumflow.checkpoint import save_checkpoint
from quantumflow.ot_cfm import (
    cfm_loss_step,
    compute_eigenvalue_spread,
    compute_mode_metrics_3d,
    empirical_wasserstein_distance,
    exact_ot_coupling,
    init_potential_network,
    integrate_ode,
    sample_cube_gaussians_3d,
    save_model_params,
    sliced_wasserstein_distance,
)


def _atomic_write_json(destination: Path, payload: dict, *, indent: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            json.dump(payload, temporary, allow_nan=False, indent=indent)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def train_and_eval_3d(
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
    """Train and evaluate 3D OT-CFM with Isotropic-Hessian regularization."""
    if num_steps < 1 or batch_size < 1 or eval_samples < 1:
        raise ValueError("num_steps, batch_size, and eval_samples must be positive")
    if not np.isfinite(beta) or beta < 0 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("beta must be finite and non-negative; lr must be finite and positive")

    step_offset = step_offset or 0
    if sample_offset is None:
        sample_offset = step_offset * batch_size

    key = jax.random.key(seed)
    numpy_rng = np.random.default_rng(seed)
    key, subkey = jax.random.split(key)

    # 3D Potential network: input dimension is 3
    params = init_potential_network(subkey, in_dim=3, hidden_dims=(128, 128, 128))
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
        x0_raw = numpy_rng.standard_normal((batch_size, 3))
        x1_raw = np.asarray(sample_cube_gaussians_3d(k_x1, batch_size))
        x0_ot, x1_ot = exact_ot_coupling(x0_raw, x1_raw)
        new_p, new_opt_s, aux = update_step(
            p,
            opt_s,
            t,
            jnp.array(x0_ot, dtype=jnp.float32),
            jnp.array(x1_ot, dtype=jnp.float32),
        )
        return new_p, new_opt_s, next_key, aux

    history = []
    print(
        f"Starting 3D OT-CFM training with beta={beta}, "
        f"num_steps={num_steps}, batch_size={batch_size}"
    )
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
                dimension=3,
                progress_unit="samples",
            )
            values = {
                "loss": loss,
                "loss_cfm": loss_cfm,
                "loss_iso": loss_iso,
                "optimizer_step": absolute_optimizer_step,
                "samples_seen": absolute_samples,
            }
            history.append([time.time(), absolute_samples, values])
            if output_dir:
                progress = {
                    "history": history,
                    "step": absolute_samples,
                    "total": sample_offset + num_steps * batch_size,
                    "values": {**values, "progress_unit": "samples"},
                }
                _atomic_write_json(output_dir / "progress.json", progress)
            print(
                f"[3D OT-CFM {step}/{num_steps}] loss: {loss:.5f} "
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
                "dimension": 3,
            },
        )
    print(f"Training completed in {training_seconds:.2f}s. Evaluating in 3D...")
    evaluation_start = time.time()

    # 3D Evaluation
    if fixed_eval_seed is not None:
        eval_seed = fixed_eval_seed
        projection_seed = fixed_eval_seed + 1_000_000
    else:
        eval_seed = seed + 1_000_003
        projection_seed = seed + 2_000_003

    k_eval_x0 = jax.random.key(eval_seed)
    k_eval_target = jax.random.key(eval_seed + 1)
    k_projection = jax.random.key(projection_seed)
    x0_eval = jax.random.normal(k_eval_x0, shape=(eval_samples, 3))
    target_samples = sample_cube_gaussians_3d(k_eval_target, eval_samples)

    generated_samples, trajectory = integrate_ode(params, x0_eval, num_steps=50)
    sliced_distance = float(
        sliced_wasserstein_distance(
            generated_samples, target_samples, key=k_projection, num_projections=128
        )
    )
    wasserstein_distance = empirical_wasserstein_distance(generated_samples, target_samples)
    mode_metrics = compute_mode_metrics_3d(generated_samples)

    # 3D Eigenvalue spread along trajectory
    mean_spreads = []
    max_spreads = []
    for t_idx, x_t in enumerate(trajectory):
        t_val = t_idx / (len(trajectory) - 1)
        spread_dict = compute_eigenvalue_spread(params, x_t[:128], t=t_val)
        mean_spreads.append(float(spread_dict["mean"]))
        max_spreads.append(float(spread_dict["max"]))
    mean_eig_spread = float(np.mean(mean_spreads))
    max_eig_spread = float(np.max(max_spreads))

    # ODE Pareto step-count evaluation in 3D
    step_counts = [5, 10, 20, 50]
    pareto_sliced = {}
    pareto_w2 = {}
    for sc in step_counts:
        gen_sc, _ = integrate_ode(params, x0_eval, num_steps=sc)
        swd_sc = float(
            sliced_wasserstein_distance(
                gen_sc, target_samples, key=k_projection, num_projections=128
            )
        )
        w2_sc = float(empirical_wasserstein_distance(gen_sc, target_samples))
        pareto_sliced[str(sc)] = swd_sc
        pareto_w2[str(sc)] = w2_sc

    evaluation_seconds = time.time() - evaluation_start

    metrics = {
        "schema_version": 2,
        "seed": seed,
        "dimension": 3,
        "config": {
            "beta": beta,
            "num_steps": num_steps,
            "batch_size": batch_size,
            "learning_rate": lr,
            "eval_samples": eval_samples,
            "ode_step_counts": step_counts,
        },
        "beta": beta,
        "final_loss": float(aux["loss"]),
        "final_loss_cfm": float(aux["loss_cfm"]),
        "final_loss_iso": float(aux["loss_iso"]),
        "sliced_wasserstein_distance": sliced_distance,
        "empirical_wasserstein_distance": wasserstein_distance,
        "mean_eigenvalue_spread": mean_eig_spread,
        "max_eigenvalue_spread": max_eig_spread,
        "modes_covered": mode_metrics["modes_covered"],
        "missing_modes": mode_metrics["missing_modes"],
        "mode_counts": mode_metrics["mode_counts"],
        "mode_entropy": mode_metrics["mode_entropy"],
        "pareto_step_sliced_wasserstein_distance": pareto_sliced,
        "pareto_step_empirical_wasserstein_distance": pareto_w2,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "elapsed_seconds": training_seconds + evaluation_seconds,
        "provenance": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "optax": optax.__version__,
            "device": str(jax.devices()[0]),
            "exp_name": os.environ.get("EXP_NAME"),
            "exp_sweep": os.environ.get("EXP_SWEEP"),
        },
    }

    if output_dir:
        output_path = Path(output_dir)
        _atomic_write_json(output_path / "metrics_3d.json", metrics, indent=2)
        print(f"Saved 3D metrics to {output_path / 'metrics_3d.json'}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 3D OT-CFM with Isotropic-Hessian regularization."
    )
    parser.add_argument("--beta", type=float, default=0.0, help="Penalty weight beta")
    parser.add_argument("--steps", type=int, default=800, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Experiment seed")
    parser.add_argument("--eval-samples", type=int, default=1024, help="Evaluation samples")
    args = parser.parse_args()

    train_and_eval_3d(
        beta=args.beta,
        num_steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
        eval_samples=args.eval_samples,
    )


if __name__ == "__main__":
    main()
