"""Experiment runner for multi-dimensional (>1D) convex potential-to-orbital energy mapping."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from quantumflow import expdash
from quantumflow.jax.convex import (
    functional_derivative,
    init_icnn,
    kinetic_energy,
    make_composite_training_step,
    reconstruct_potential,
    solve_ground_state_density,
)
from quantumflow.multidim import (
    Grid,
    generate_multidim_dataset,
    solve_multidim_schroedinger,
)


def run_experiment(
    dimension: int = 2,
    grid_points: int = 24,
    grid_bound: float = 3.5,
    num_orbitals: int = 3,
    train_size: int = 200,
    test_size: int = 50,
    hidden_units: tuple[int, ...] = (128, 128),
    num_steps: int = 800,
    batch_size: int = 32,
    lr: float = 2e-3,
    alpha_derivative: float = 1.0,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Train and evaluate convex functional for multi-dimensional orbital energy mapping."""
    if dimension not in (1, 2, 3):
        raise ValueError("dimension must be 1, 2, or 3")
    if num_steps < 1 or train_size < 1 or test_size < 1:
        raise ValueError("num_steps, train_size, and test_size must be positive")

    print(f"=== Starting {dimension}D Convex Potential-to-Orbital Energy Mapping Experiment ===")
    print(
        f"Dimension: {dimension}D | Grid: {grid_points}^{dimension} = "
        f"{grid_points**dimension} pts | Orbitals: {num_orbitals}"
    )
    print(
        f"Train size: {train_size} | Test size: {test_size} | "
        f"Steps: {num_steps} | Batch: {batch_size} | LR: {lr}"
    )

    grid = Grid.create(
        dimension=dimension,
        lower=-grid_bound,
        upper=grid_bound,
        points=grid_points,
    )

    # 1. Dataset Generation
    data_start = time.time()
    print("Generating training dataset...")
    train_data = generate_multidim_dataset(
        grid=grid,
        dataset_size=train_size,
        num_orbitals=num_orbitals,
        seed=seed,
    )
    print("Generating test dataset...")
    test_data = generate_multidim_dataset(
        grid=grid,
        dataset_size=test_size,
        num_orbitals=num_orbitals,
        seed=seed + 9999,
    )
    data_seconds = time.time() - data_start
    print(f"Datasets generated in {data_seconds:.2f}s.")

    # 2. Model Initialization & Training
    train_start = time.time()
    key = jax.random.key(seed)
    key, key_init = jax.random.split(key)
    params = init_icnn(key_init, input_size=grid.total_points, hidden_units=hidden_units)

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)
    step_fn = make_composite_training_step(
        optimizer,
        alpha_derivative=alpha_derivative,
        volume_element=grid.volume_element,
    )

    density_train_jnp = jnp.asarray(train_data["densities"], dtype=jnp.float32)
    t_train_jnp = jnp.asarray(train_data["kinetic_energies"], dtype=jnp.float32)
    deriv_train_jnp = jnp.asarray(train_data["derivatives"], dtype=jnp.float32)

    rng = np.random.default_rng(seed)

    print("Training ICNN functional...")
    for step in range(1, num_steps + 1):
        idx = rng.choice(train_size, size=batch_size, replace=False)
        batch_density = density_train_jnp[idx]
        batch_t = t_train_jnp[idx]
        batch_deriv = deriv_train_jnp[idx]

        params, opt_state, aux = step_fn(
            params,
            opt_state,
            batch_density,
            batch_t,
            batch_deriv,
        )

        if step % 50 == 0 or step == num_steps:
            loss_val = float(aux["loss"])
            loss_e = float(aux["loss_energy"])
            loss_d = float(aux["loss_derivative"])
            expdash.report(
                step=step,
                total=num_steps,
                loss=loss_val,
                loss_energy=loss_e,
                loss_derivative=loss_d,
                penalty=loss_d,
            )
            print(
                f"[{step:4d}/{num_steps:4d}] Loss: {loss_val:.6f} "
                f"(Energy MSE: {loss_e:.6f}, Derivative MSE: {loss_d:.6f})"
            )

    train_seconds = time.time() - train_start
    print(f"Training completed in {train_seconds:.2f}s.")

    # 3. Evaluation on Test Set
    eval_start = time.time()
    print("Evaluating on test set...")
    density_test_jnp = jnp.asarray(test_data["densities"], dtype=jnp.float32)
    t_test_true = test_data["kinetic_energies"]
    v_test_true = test_data["potentials"]
    mu_test_true = test_data["chemical_potentials"]
    eps_test_true = test_data["orbital_energies"]

    # (a) Kinetic energy predictions
    t_test_pred = np.asarray(kinetic_energy(params, density_test_jnp))
    t_mae = float(np.mean(np.abs(t_test_pred - t_test_true)))
    t_rmse = float(np.sqrt(np.mean((t_test_pred - t_test_true) ** 2)))

    # (b) Potential & Chemical potential reconstruction: v_pred = mu - delta T / delta n
    pred_derivatives = np.asarray(
        functional_derivative(params, density_test_jnp, volume_element=grid.volume_element)
    )
    # Chemical potential estimation from density (HOMO energy)
    # Using the dual relation mu_pred ~ mean(delta T/delta n + v)
    mu_pred_from_density = np.mean(pred_derivatives + v_test_true, axis=1)
    mu_mae = float(np.mean(np.abs(mu_pred_from_density - mu_test_true)))

    # Reconstructed potentials
    v_test_pred = np.zeros_like(v_test_true)
    orbital_energies_pred = np.zeros_like(eps_test_true)

    for i in range(test_size):
        v_pred_i = np.asarray(
            reconstruct_potential(
                params,
                density_test_jnp[i],
                mu_test_true[i],
                volume_element=grid.volume_element,
            )
        )
        v_test_pred[i] = v_pred_i

        # Solve Schrödinger equation on reconstructed potential to recover orbital spectrum
        sol_rec = solve_multidim_schroedinger(
            potential=v_pred_i,
            grid=grid,
            num_orbitals=num_orbitals,
        )
        orbital_energies_pred[i] = sol_rec["orbital_energies"]

    v_mae = float(np.mean(np.abs(v_test_pred - v_test_true)))
    v_rmse = float(np.sqrt(np.mean((v_test_pred - v_test_true) ** 2)))

    # Orbital energy errors for each state epsilon_0, ..., epsilon_{N-1}
    orbital_mae_per_state = [
        float(np.mean(np.abs(orbital_energies_pred[:, k] - eps_test_true[:, k])))
        for k in range(num_orbitals)
    ]
    mean_orbital_mae = float(np.mean(orbital_mae_per_state))

    # (c) Variational Ground-State Inversion: v(r) -> n*(r) -> orbital energies
    print("Testing variational density optimization from potential v(r)...")
    var_eval_count = min(10, test_size)
    var_density_errors = []
    var_mu_errors = []
    var_orbital_errors = []

    for i in range(var_eval_count):
        var_sol = solve_ground_state_density(
            params=params,
            potential=v_test_true[i],
            num_particles=float(num_orbitals),
            volume_element=grid.volume_element,
            num_steps=250,
            lr=3e-2,
        )
        n_opt = np.asarray(var_sol["density"])
        mu_opt = float(var_sol["chemical_potential"])
        var_density_errors.append(float(np.mean(np.abs(n_opt - test_data["densities"][i]))))
        var_mu_errors.append(float(np.abs(mu_opt - mu_test_true[i])))

        # Reconstruct potential from variational density & get orbital energies
        v_opt = np.asarray(reconstruct_potential(params, n_opt, mu_opt))
        sol_var_rec = solve_multidim_schroedinger(
            potential=v_opt,
            grid=grid,
            num_orbitals=num_orbitals,
        )
        var_orbital_errors.append(
            float(np.mean(np.abs(sol_var_rec["orbital_energies"] - eps_test_true[i])))
        )

    eval_seconds = time.time() - eval_start
    total_seconds = data_seconds + train_seconds + eval_seconds

    metrics = {
        "dimension": dimension,
        "grid_points_per_dim": grid_points,
        "total_grid_points": grid.total_points,
        "num_orbitals": num_orbitals,
        "train_size": train_size,
        "test_size": test_size,
        "num_steps": num_steps,
        "kinetic_energy_mae": t_mae,
        "kinetic_energy_rmse": t_rmse,
        "potential_mae": v_mae,
        "potential_rmse": v_rmse,
        "chemical_potential_mae": mu_mae,
        "mean_orbital_energy_mae": mean_orbital_mae,
        "orbital_energy_mae_per_state": orbital_mae_per_state,
        "variational_density_mae": float(np.mean(var_density_errors)),
        "variational_chemical_potential_mae": float(np.mean(var_mu_errors)),
        "variational_orbital_energy_mae": float(np.mean(var_orbital_errors)),
        "data_seconds": data_seconds,
        "training_seconds": train_seconds,
        "evaluation_seconds": eval_seconds,
        "total_seconds": total_seconds,
    }

    print("\n" + "=" * 60)
    print(f"=== {dimension}D EXPERIMENT RESULTS SUMMARY ===")
    print(f"Kinetic Energy MAE:           {t_mae:.5f} Hartree")
    print(f"Potential Reconstruction MAE: {v_mae:.5f} Hartree")
    print(f"Chemical Potential (HOMO) MAE: {mu_mae:.5f} Hartree")
    print(f"Mean Orbital Energy MAE:      {mean_orbital_mae:.5f} Hartree")
    for k, mae_k in enumerate(orbital_mae_per_state):
        print(f"  - Orbital #{k + 1} (eps_{k}) MAE:   {mae_k:.5f} Hartree")
    print(f"Variational Density Inversion MAE: {np.mean(var_density_errors):.6f}")
    print(f"Variational Orbital Energy MAE:   {np.mean(var_orbital_errors):.5f} Hartree")
    print("=" * 60 + "\n")

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / f"metrics_{dimension}d.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved results to {output_path / f'metrics_{dimension}d.json'}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-dimensional convex potential-to-orbital energy mapping experiments."
    )
    parser.add_argument(
        "--dimension", type=int, default=2, choices=[1, 2, 3], help="Spatial dimension (1, 2, or 3)"
    )
    parser.add_argument("--grid-points", type=int, default=24, help="Grid points per dimension")
    parser.add_argument("--orbitals", type=int, default=3, help="Number of occupied orbitals")
    parser.add_argument("--train-size", type=int, default=200, help="Training dataset size")
    parser.add_argument("--test-size", type=int, default=50, help="Test dataset size")
    parser.add_argument("--steps", type=int, default=800, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Derivative loss weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/convex_multidim"), help="Output directory"
    )
    args = parser.parse_args()

    run_experiment(
        dimension=args.dimension,
        grid_points=args.grid_points,
        num_orbitals=args.orbitals,
        train_size=args.train_size,
        test_size=args.test_size,
        num_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_derivative=args.alpha,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
