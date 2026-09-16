"""CLI tool to render publication-ready figures for QuantumFlow experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from quantumflow.multidim import Grid, generate_multidim_dataset, solve_multidim_schroedinger
from quantumflow.viz import (
    plot_potential_and_orbitals_1d,
    plot_reconstruction_comparison,
    plot_system_2d,
)


def render_all_figures(output_dir: Path = Path("outputs/figures")) -> None:
    """Render and save a complete set of 1D and 2D figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rendering figures to {output_dir.resolve()}...")

    # 1. 1D Potential and Orbital Energy Diagram
    grid_1d = Grid.create(dimension=1, lower=-4.0, upper=4.0, points=128)
    data_1d = generate_multidim_dataset(grid=grid_1d, dataset_size=1, num_orbitals=4, seed=42)
    v_1d = data_1d["potentials"][0]
    sol_1d = solve_multidim_schroedinger(potential=v_1d, grid=grid_1d, num_orbitals=4)

    fig1 = plot_potential_and_orbitals_1d(
        potential=v_1d,
        wavefunctions=sol_1d["wavefunctions"],
        energies=sol_1d["orbital_energies"],
        x=grid_1d.coords_1d[0],
        output_path=output_dir / "figure_1d_orbitals.png",
    )
    fig1.savefig(output_dir / "figure_1d_orbitals.svg", bbox_inches="tight")

    # 2. 1D Reconstruction & Spectrum Comparison
    # Simulate a small reconstruction perturbation for illustration
    v_rec_1d = v_1d + 0.15 * np.cos(np.linspace(0, 3 * np.pi, len(v_1d)))
    sol_rec_1d = solve_multidim_schroedinger(potential=v_rec_1d, grid=grid_1d, num_orbitals=4)

    plot_reconstruction_comparison(
        v_true=v_1d,
        v_pred=v_rec_1d,
        n_true=sol_1d["density"],
        eps_true=sol_1d["orbital_energies"],
        eps_pred=sol_rec_1d["orbital_energies"],
        x=grid_1d.coords_1d[0],
        output_path=output_dir / "figure_1d_reconstruction.png",
    )

    # 3. 2D Quantum System: Potential, Electron Density, and Orbital Wavefunctions
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
        output_path=output_dir / "figure_2d_system.png",
    )
    fig2.savefig(output_dir / "figure_2d_system.svg", bbox_inches="tight")

    print(f"All figures successfully generated in {output_dir}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render publication figures for QuantumFlow.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Target directory for rendered figures",
    )
    args = parser.parse_args()
    render_all_figures(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
