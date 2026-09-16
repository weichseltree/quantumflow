from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from quantumflow.multidim import Grid, solve_multidim_schroedinger
from quantumflow.viz import (
    export_grove_exhibition_manifest,
    plot_potential_and_orbitals_1d,
    plot_reconstruction_comparison,
    plot_system_2d,
    render_density_optimization_animation,
    render_transport_flow_animation,
)


def test_plot_potential_and_orbitals_1d(tmp_path: Path) -> None:
    grid = Grid.create(dimension=1, lower=-3.0, upper=3.0, points=32)
    coords = grid.coordinates.ravel()
    v = 0.5 * coords**2
    sol = solve_multidim_schroedinger(potential=v, grid=grid, num_orbitals=2)

    out_file = tmp_path / "plot_1d.png"
    fig = plot_potential_and_orbitals_1d(
        potential=v,
        wavefunctions=sol["wavefunctions"],
        energies=sol["orbital_energies"],
        x=grid.coords_1d[0],
        output_path=out_file,
    )
    assert fig is not None
    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_plot_reconstruction_comparison(tmp_path: Path) -> None:
    grid = Grid.create(dimension=1, lower=-3.0, upper=3.0, points=32)
    coords = grid.coordinates.ravel()
    v_true = 0.5 * coords**2
    v_pred = v_true + 0.1
    sol = solve_multidim_schroedinger(potential=v_true, grid=grid, num_orbitals=2)

    out_file = tmp_path / "reconstruction.png"
    fig = plot_reconstruction_comparison(
        v_true=v_true,
        v_pred=v_pred,
        n_true=sol["density"],
        eps_true=sol["orbital_energies"],
        eps_pred=sol["orbital_energies"] + 0.1,
        x=grid.coords_1d[0],
        output_path=out_file,
    )
    assert fig is not None
    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_plot_system_2d(tmp_path: Path) -> None:
    grid = Grid.create(dimension=2, lower=-2.5, upper=2.5, points=12)
    coords = grid.coordinates
    v = 0.5 * np.sum(coords**2, axis=-1)
    sol = solve_multidim_schroedinger(potential=v, grid=grid, num_orbitals=2)

    out_file = tmp_path / "system_2d.png"
    fig = plot_system_2d(
        grid=grid,
        potential=v,
        density=sol["density"],
        wavefunctions=sol["wavefunctions"],
        energies=sol["orbital_energies"],
        output_path=out_file,
    )
    assert fig is not None
    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_render_density_optimization_animation(tmp_path: Path) -> None:
    n_target = np.array([0.1, 0.4, 0.8, 0.4, 0.1])
    traj = [n_target * (i + 1) / 5 for i in range(5)]
    energies = [5.0 - i * 0.8 for i in range(5)]
    out_gif = tmp_path / "test_density.gif"

    anim = render_density_optimization_animation(
        density_trajectory=traj,
        energy_trajectory=energies,
        target_density=n_target,
        output_path=out_gif,
        fps=5,
    )
    assert anim is not None
    assert out_gif.is_file()
    assert out_gif.stat().st_size > 0


def test_render_transport_flow_animation(tmp_path: Path) -> None:
    num_steps = 6
    num_particles = 20
    traj_pts = np.random.randn(num_steps, num_particles, 2)
    target_samples = np.random.randn(50, 2)
    out_gif = tmp_path / "test_flow.gif"

    anim = render_transport_flow_animation(
        trajectory_points=traj_pts,
        target_samples=target_samples,
        output_path=out_gif,
        fps=5,
    )
    assert anim is not None
    assert out_gif.is_file()
    assert out_gif.stat().st_size > 0


def test_export_grove_exhibition_manifest(tmp_path: Path) -> None:
    manifest = export_grove_exhibition_manifest(output_dir=tmp_path, grid_points=8, num_orbitals=2)
    assert manifest["title"]
    assert len(manifest["rooms"]) == 4
    assert (tmp_path / "grove_exhibition_manifest.json").is_file()
    assert (tmp_path / "density_3d.npy").is_file()
