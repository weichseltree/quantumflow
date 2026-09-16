"""CLI to build and export the 3D Grove exhibition for weichseltree.com/grove."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from quantumflow.jax.convex import init_icnn, kinetic_energy
from quantumflow.multidim import Grid, generate_multidim_dataset, solve_multidim_schroedinger
from quantumflow.ot_cfm import init_potential_network, integrate_ode, sample_8gaussians
from quantumflow.viz import (
    export_grove_exhibition_manifest,
    plot_potential_and_orbitals_1d,
    plot_reconstruction_comparison,
    plot_system_2d,
    render_density_optimization_animation,
    render_transport_flow_animation,
)


def build_full_grove_exhibition(output_dir: Path = Path("outputs/grove_exhibition")) -> None:
    """Build all figures, animations, shaders, and metadata for the Grove 3D gallery."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    videos_dir = output_dir / "videos"
    figures_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    print("=== Building Grove 3D QuantumFlow Exhibition ===")

    # 1. Generate Figures
    grid_1d = Grid.create(dimension=1, lower=-4.0, upper=4.0, points=128)
    data_1d = generate_multidim_dataset(grid=grid_1d, dataset_size=1, num_orbitals=4, seed=42)
    v_1d = data_1d["potentials"][0]
    sol_1d = solve_multidim_schroedinger(potential=v_1d, grid=grid_1d, num_orbitals=4)

    fig1 = plot_potential_and_orbitals_1d(
        potential=v_1d,
        wavefunctions=sol_1d["wavefunctions"],
        energies=sol_1d["orbital_energies"],
        x=grid_1d.coords_1d[0],
        output_path=figures_dir / "figure_1d_orbitals.png",
    )
    fig1.savefig(figures_dir / "figure_1d_orbitals.svg", bbox_inches="tight")

    v_rec_1d = v_1d + 0.15 * np.cos(np.linspace(0, 3 * np.pi, len(v_1d)))
    sol_rec_1d = solve_multidim_schroedinger(potential=v_rec_1d, grid=grid_1d, num_orbitals=4)
    plot_reconstruction_comparison(
        v_true=v_1d,
        v_pred=v_rec_1d,
        n_true=sol_1d["density"],
        eps_true=sol_1d["orbital_energies"],
        eps_pred=sol_rec_1d["orbital_energies"],
        x=grid_1d.coords_1d[0],
        output_path=figures_dir / "figure_1d_reconstruction.png",
    )

    grid_2d = Grid.create(dimension=2, lower=-3.5, upper=3.5, points=32)
    data_2d = generate_multidim_dataset(grid=grid_2d, dataset_size=1, num_orbitals=3, seed=101)
    v_2d = data_2d["potentials"][0]
    sol_2d = solve_multidim_schroedinger(potential=v_2d, grid=grid_2d, num_orbitals=3)
    fig2 = plot_system_2d(
        grid=grid_2d,
        potential=v_2d,
        density=sol_2d["density"],
        wavefunctions=sol_2d["wavefunctions"],
        energies=sol_2d["orbital_energies"],
        output_path=figures_dir / "figure_2d_system.png",
    )
    fig2.savefig(figures_dir / "figure_2d_system.svg", bbox_inches="tight")

    # 2. Generate Density Relaxation Video
    params = init_icnn(jax.random.key(123), input_size=grid_1d.total_points, hidden_units=(32, 32))
    v_arr = jnp.asarray(v_1d, dtype=jnp.float32)
    dv = grid_1d.volume_element
    num_particles = 4.0
    logits = jnp.zeros((grid_1d.total_points,), dtype=jnp.float32)
    optimizer = optax.adam(learning_rate=0.08)
    opt_state = optimizer.init(logits)

    def logits_to_density(log_arr: jax.Array) -> jax.Array:
        sp = jax.nn.softplus(log_arr)
        return (sp / (jnp.sum(sp) * dv)) * num_particles

    def variational_energy(log_arr: jax.Array) -> jax.Array:
        n = logits_to_density(log_arr)
        return kinetic_energy(params, n) + jnp.sum(v_arr * n) * dv

    density_trajectory = []
    energy_trajectory = []
    for _ in range(40):
        density_trajectory.append(np.asarray(logits_to_density(logits)))
        energy_trajectory.append(float(variational_energy(logits)))
        grads = jax.grad(variational_energy)(logits)
        updates, opt_state = optimizer.update(grads, opt_state, logits)
        logits = optax.apply_updates(logits, updates)

    render_density_optimization_animation(
        density_trajectory=density_trajectory,
        energy_trajectory=energy_trajectory,
        target_density=sol_1d["density"],
        x=grid_1d.coords_1d[0],
        output_path=videos_dir / "density_relaxation.gif",
        fps=12,
    )

    # 3. Generate OT Transport Video
    key = jax.random.key(42)
    key_params, key_x0, key_target = jax.random.split(key, 3)
    net_params = init_potential_network(key_params, in_dim=2, hidden_dims=(64, 64))
    x0 = jax.random.normal(key_x0, shape=(500, 2))
    target = np.asarray(sample_8gaussians(key_target, batch_size=1000))
    _, trajectory = integrate_ode(net_params, x0, num_steps=40)
    render_transport_flow_animation(
        trajectory_points=np.asarray(trajectory),
        target_samples=target,
        output_path=videos_dir / "transport_flow.gif",
        fps=15,
    )

    # 4. Export Manifest & 3D Shaders
    export_grove_exhibition_manifest(output_dir=output_dir, grid_points=16, num_orbitals=3)

    print(f"\nGrove Exhibition build complete in {output_dir.resolve()}!")
    print("Ready to integrate into weichseltree.com/grove.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and export 3D Grove exhibition assets for QuantumFlow."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/grove_exhibition"),
        help="Target output directory",
    )
    args = parser.parse_args()
    build_full_grove_exhibition(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
