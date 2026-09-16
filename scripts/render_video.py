"""CLI tool to render animated videos and dynamic visualizations for QuantumFlow."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from quantumflow.jax.convex import init_icnn, kinetic_energy
from quantumflow.multidim import Grid, generate_multidim_dataset
from quantumflow.ot_cfm import (
    init_potential_network,
    integrate_ode,
    sample_8gaussians,
)
from quantumflow.viz import (
    render_density_optimization_animation,
    render_transport_flow_animation,
)


def render_density_relaxation_video(
    output_dir: Path = Path("outputs/videos"),
    format_type: str = "gif",
    num_steps: int = 40,
) -> None:
    """Simulate and render variational density relaxation trajectory."""
    print("Generating variational ground-state density relaxation video...")
    grid = Grid.create(dimension=1, lower=-4.0, upper=4.0, points=64)
    data = generate_multidim_dataset(grid=grid, dataset_size=1, num_orbitals=3, seed=42)
    v_true = data["potentials"][0]
    n_target = data["densities"][0]

    # Initialize a small ICNN
    params = init_icnn(jax.random.key(123), input_size=grid.total_points, hidden_units=(32, 32))
    v_arr = jnp.asarray(v_true, dtype=jnp.float32)
    dv = grid.volume_element
    num_particles = 3.0

    logits = jnp.zeros((grid.total_points,), dtype=jnp.float32)
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

    for _ in range(num_steps):
        curr_n = np.asarray(logits_to_density(logits))
        curr_e = float(variational_energy(logits))
        density_trajectory.append(curr_n)
        energy_trajectory.append(curr_e)

        grads = jax.grad(variational_energy)(logits)
        updates, opt_state = optimizer.update(grads, opt_state, logits)
        logits = optax.apply_updates(logits, updates)

    output_path = output_dir / f"density_relaxation.{format_type}"
    render_density_optimization_animation(
        density_trajectory=density_trajectory,
        energy_trajectory=energy_trajectory,
        target_density=n_target,
        x=grid.coords_1d[0],
        output_path=output_path,
        fps=12,
    )


def render_ot_transport_video(
    output_dir: Path = Path("outputs/videos"),
    format_type: str = "gif",
    num_particles: int = 500,
    ode_steps: int = 40,
) -> None:
    """Render 2D continuous normalizing flow / Optimal Transport trajectory video."""
    print("Generating Optimal Transport flow matching particle animation...")
    key = jax.random.key(42)
    key_params, key_x0, key_target = jax.random.split(key, 3)

    params = init_potential_network(key_params, in_dim=2, hidden_dims=(64, 64))
    x0 = jax.random.normal(key_x0, shape=(num_particles, 2))
    target = np.asarray(sample_8gaussians(key_target, batch_size=1000))

    _, trajectory = integrate_ode(params, x0, num_steps=ode_steps)
    trajectory_np = np.asarray(trajectory)  # shape: (num_steps + 1, num_particles, 2)

    output_path = output_dir / f"transport_flow.{format_type}"
    render_transport_flow_animation(
        trajectory_points=trajectory_np,
        target_samples=target,
        output_path=output_path,
        fps=15,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render animated videos and visualizations for QuantumFlow."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/videos"),
        help="Directory to save generated animations",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gif",
        choices=["gif", "mp4"],
        help="Output video format (gif or mp4)",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="all",
        choices=["all", "density", "transport"],
        help="Type of animation to render",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.type in ("all", "density"):
        render_density_relaxation_video(output_dir=args.output_dir, format_type=args.format)
    if args.type in ("all", "transport"):
        render_ot_transport_video(output_dir=args.output_dir, format_type=args.format)


if __name__ == "__main__":
    main()
