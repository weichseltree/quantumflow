"""Visualization and animation tools for QuantumFlow experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from quantumflow.multidim import Grid, solve_multidim_schroedinger

plt.rcParams["svg.fonttype"] = "none"


def plot_potential_and_orbitals_1d(
    potential: np.ndarray,
    wavefunctions: np.ndarray,
    energies: np.ndarray,
    x: np.ndarray,
    num_orbitals: int | None = None,
    title: str = "1D Potential and Orbital Energy Levels",
    figsize: tuple[float, float] = (10, 6),
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Render a publication-ready 1D potential with energy levels and orbital densities."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(x, potential, "k-", linewidth=2.0, label="Potential $v(x)$")

    k_max = num_orbitals or wavefunctions.shape[1]
    cmap = plt.get_cmap("tab10")

    for i in range(k_max):
        color = cmap(i % 10)
        e_i = float(energies[i])
        ax.axhline(
            e_i,
            color=color,
            linestyle="--",
            alpha=0.7,
            label=f"$\\epsilon_{i} = {e_i:.3f}$ Ha",
        )
        density_i = wavefunctions[:, i] ** 2
        scale = (np.max(energies) - np.min(potential)) * 0.15 / (np.max(density_i) + 1e-8)
        ax.plot(x, e_i + density_i * scale, color=color, linewidth=1.5)
        ax.fill_between(x, e_i, e_i + density_i * scale, color=color, alpha=0.25)

    ax.set_xlabel("Position $x$ [Bohr]", fontsize=12)
    ax.set_ylabel("Energy [Hartree]", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def plot_reconstruction_comparison(
    v_true: np.ndarray,
    v_pred: np.ndarray,
    n_true: np.ndarray,
    eps_true: np.ndarray,
    eps_pred: np.ndarray,
    x: np.ndarray | None = None,
    title: str = "Convex Functional Potential & Energy Reconstruction",
    figsize: tuple[float, float] = (14, 5),
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Multi-panel comparison of true vs predicted potentials, densities, and orbital spectrum."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    x_axis = np.arange(len(v_true)) if x is None else x

    # Panel 1: Potential reconstruction
    ax1.plot(x_axis, v_true, "k-", linewidth=2.0, label="True $v(r)$")
    ax1.plot(x_axis, v_pred, "r--", linewidth=1.8, label="Reconstructed $v_{\\mathrm{pred}}(r)$")
    ax1.set_title("Potential Reconstruction", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Grid Coordinate", fontsize=11)
    ax1.set_ylabel("Energy [Hartree]", fontsize=11)
    ax1.legend(frameon=True, fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Electron density
    ax2.plot(x_axis, n_true, "b-", linewidth=2.0, label="Density $n(r)$")
    ax2.fill_between(x_axis, 0, n_true, color="b", alpha=0.2)
    ax2.set_title("Electron Density", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Grid Coordinate", fontsize=11)
    ax2.set_ylabel("Density $n(r)$", fontsize=11)
    ax2.legend(frameon=True, fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Orbital energy spectrum comparison
    indices = np.arange(len(eps_true))
    width = 0.35
    ax3.bar(
        indices - width / 2,
        eps_true,
        width,
        label="True $\\epsilon_i$",
        color="black",
        alpha=0.75,
    )
    ax3.bar(
        indices + width / 2,
        eps_pred,
        width,
        label="Pred $\\epsilon_i$",
        color="crimson",
        alpha=0.75,
    )
    ax3.set_xticks(indices)
    ax3.set_xticklabels([f"$\\epsilon_{i}$" for i in indices])
    ax3.set_title("Orbital Energy Spectrum", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Energy [Hartree]", fontsize=11)
    ax3.legend(frameon=True, fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def plot_system_2d(
    grid: Grid,
    potential: np.ndarray,
    density: np.ndarray,
    wavefunctions: np.ndarray | None = None,
    energies: np.ndarray | None = None,
    num_orbitals_preview: int = 3,
    title: str = "2D Quantum System (Potential, Density, & Orbitals)",
    figsize: tuple[float, float] = (15, 8),
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Render 2D spatial contour plots for potential, total density, and individual orbitals."""
    if grid.dimension != 2:
        raise ValueError(f"plot_system_2d requires 2D grid, got {grid.dimension}D")

    nx, ny = grid.points_per_dim
    extent = [grid.bounds[0][0], grid.bounds[0][1], grid.bounds[1][0], grid.bounds[1][1]]

    num_panels = 2 + (num_orbitals_preview if wavefunctions is not None else 0)
    cols = min(num_panels, 4)
    rows = int(np.ceil(num_panels / cols))

    fig, axs = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    axs_flat = np.array(axs).ravel()

    # 1. Potential
    v_2d = potential.reshape(nx, ny)
    im0 = axs_flat[0].imshow(v_2d.T, origin="lower", extent=extent, cmap="viridis")
    axs_flat[0].set_title("Potential $v(x, y)$", fontweight="bold")
    axs_flat[0].set_xlabel("$x$ [Bohr]")
    axs_flat[0].set_ylabel("$y$ [Bohr]")
    fig.colorbar(im0, ax=axs_flat[0], fraction=0.046, pad=0.04)

    # 2. Total Density
    n_2d = density.reshape(nx, ny)
    im1 = axs_flat[1].imshow(n_2d.T, origin="lower", extent=extent, cmap="inferno")
    axs_flat[1].set_title("Electron Density $n(x, y)$", fontweight="bold")
    axs_flat[1].set_xlabel("$x$ [Bohr]")
    axs_flat[1].set_ylabel("$y$ [Bohr]")
    fig.colorbar(im1, ax=axs_flat[1], fraction=0.046, pad=0.04)

    # 3. Individual Orbitals
    if wavefunctions is not None:
        for idx in range(min(num_orbitals_preview, wavefunctions.shape[1])):
            panel_idx = 2 + idx
            if panel_idx < len(axs_flat):
                psi_2d = wavefunctions[:, idx].reshape(nx, ny)
                im_k = axs_flat[panel_idx].imshow(
                    (psi_2d**2).T, origin="lower", extent=extent, cmap="magma"
                )
                e_val = f"$\\epsilon_{idx} = {energies[idx]:.3f}$" if energies is not None else ""
                axs_flat[panel_idx].set_title(
                    f"Orbital #{idx + 1} $|\\psi_{idx}|^2$ {e_val}", fontweight="bold"
                )
                axs_flat[panel_idx].set_xlabel("$x$ [Bohr]")
                axs_flat[panel_idx].set_ylabel("$y$ [Bohr]")
                fig.colorbar(im_k, ax=axs_flat[panel_idx], fraction=0.046, pad=0.04)

    # Turn off unused subplots
    for p in range(num_panels, len(axs_flat)):
        axs_flat[p].axis("off")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def render_density_optimization_animation(
    density_trajectory: Sequence[np.ndarray],
    energy_trajectory: Sequence[float],
    target_density: np.ndarray,
    x: np.ndarray | None = None,
    output_path: str | Path = "outputs/videos/density_optimization.gif",
    fps: int = 15,
    figsize: tuple[float, float] = (12, 5),
    dpi: int = 150,
) -> animation.FuncAnimation:
    """Render and export an animation of ground-state density relaxation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    frames = len(density_trajectory)
    x_axis = np.arange(len(target_density)) if x is None else x

    # Panel 1: Density evolution
    ax1.plot(
        x_axis,
        target_density,
        "k--",
        linewidth=2.0,
        label="Exact Target Density $n^*(r)$",
    )
    (line_current,) = ax1.plot(
        x_axis,
        density_trajectory[0],
        "r-",
        linewidth=2.0,
        label="Variational $n_t(r)$",
    )
    fill_current = ax1.fill_between(x_axis, 0, density_trajectory[0], color="crimson", alpha=0.3)

    ax1.set_ylim(0, float(np.max(target_density) * 1.35))
    ax1.set_xlabel("Coordinate $r$", fontsize=11)
    ax1.set_ylabel("Density $n(r)$", fontsize=11)
    ax1.set_title("Variational Ground-State Inversion", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Energy convergence curve
    energies = np.asarray(energy_trajectory)
    ax2.set_xlim(0, frames)
    ax2.set_ylim(float(np.min(energies) - 0.5), float(np.max(energies) + 0.5))
    (line_energy,) = ax2.plot([], [], "b-", linewidth=2.0, label="Energy $E[n_t]$")
    (point_energy,) = ax2.plot([], [], "ro", markersize=6)
    ax2.set_xlabel("Optimization Step", fontsize=11)
    ax2.set_ylabel("Total Energy [Hartree]", fontsize=11)
    ax2.set_title("Energy Minimization Curve", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    step_text = ax1.text(
        0.05,
        0.90,
        f"Step 0/{frames}",
        transform=ax1.transAxes,
        fontsize=11,
        fontweight="bold",
    )

    fig.tight_layout()

    def update(frame_idx: int):
        nonlocal fill_current
        current_n = density_trajectory[frame_idx]
        line_current.set_ydata(current_n)
        fill_current.remove()
        fill_current = ax1.fill_between(x_axis, 0, current_n, color="crimson", alpha=0.3)

        line_energy.set_data(np.arange(frame_idx + 1), energies[: frame_idx + 1])
        point_energy.set_data([frame_idx], [energies[frame_idx]])

        step_text.set_text(f"Step {frame_idx + 1}/{frames}")
        return line_current, line_energy, point_energy, step_text

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False, interval=1000 // fps)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".mp4":
        anim.save(str(out), writer="ffmpeg", fps=fps, dpi=dpi)
    else:
        anim.save(str(out), writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"Saved video animation to {out}")

    return anim


def render_transport_flow_animation(
    trajectory_points: np.ndarray,
    target_samples: np.ndarray,
    output_path: str | Path = "outputs/videos/transport_flow.gif",
    fps: int = 20,
    figsize: tuple[float, float] = (8, 8),
    dpi: int = 150,
) -> animation.FuncAnimation:
    """Render and export an animation of 2D OT-CFM particle transport flow."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    num_steps = len(trajectory_points)

    ax.scatter(
        target_samples[:, 0],
        target_samples[:, 1],
        c="gray",
        alpha=0.15,
        s=12,
        label="Target Distribution",
    )
    scatter_particles = ax.scatter([], [], c="royalblue", alpha=0.6, s=16, label="Particles $x(t)$")

    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_title("Optimal Transport Continuous Normalizing Flow", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.5)

    time_text = ax.text(
        0.05,
        0.93,
        "t = 0.00",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    def update(frame_idx: int):
        pts = trajectory_points[frame_idx]
        scatter_particles.set_offsets(pts)
        t_val = frame_idx / (num_steps - 1)
        time_text.set_text(f"t = {t_val:.2f}")
        return scatter_particles, time_text

    anim = animation.FuncAnimation(fig, update, frames=num_steps, blit=False, interval=1000 // fps)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".mp4":
        anim.save(str(out), writer="ffmpeg", fps=fps, dpi=dpi)
    else:
        anim.save(str(out), writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"Saved flow animation to {out}")

    return anim


def render_transport_3d_animation(
    trajectory_points: np.ndarray,
    target_samples: np.ndarray,
    output_path: str | Path = "outputs/videos/transport_3d_flow.gif",
    fps: int = 20,
    figsize: tuple[float, float] = (9, 9),
    dpi: int = 150,
) -> animation.FuncAnimation:
    """Render and export an animation of 3D OT-CFM particle transport flow towards 8-cube Gaussians."""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    num_steps = len(trajectory_points)

    # Plot target 8-cube modes as muted points
    ax.scatter(
        target_samples[:, 0],
        target_samples[:, 1],
        target_samples[:, 2],
        c="#7f8c8d",
        alpha=0.12,
        s=10,
        label="8-Cube Target Modes",
    )

    p0 = trajectory_points[0]
    scatter_particles = ax.scatter(
        p0[:, 0],
        p0[:, 1],
        p0[:, 2],
        c="#3498db",
        alpha=0.65,
        s=14,
        edgecolors="none",
        label=r"Transport Particles $\mathbf{x}(t)$",
    )

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_zlim(-3.5, 3.5)
    ax.set_xlabel("$x$", fontsize=11)
    ax.set_ylabel("$y$", fontsize=11)
    ax.set_zlabel("$z$", fontsize=11)
    ax.set_title("3D Optimal Transport Flow Matching (OT-CFM)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.view_init(elev=24, azim=38)

    time_text = fig.text(
        0.05,
        0.92,
        "t = 0.00",
        fontsize=12,
        fontweight="bold",
        color="#2c3e50",
    )
    fig.tight_layout()

    def update(frame_idx: int):
        pts = trajectory_points[frame_idx]
        scatter_particles._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        # Smoothly orbit camera during animation for 3D depth perception
        ax.view_init(
            elev=24 + 6 * np.sin(2 * np.pi * frame_idx / num_steps),
            azim=38 + 360 * frame_idx / num_steps,
        )
        t_val = frame_idx / (num_steps - 1)
        time_text.set_text(f"t = {t_val:.2f}")
        return scatter_particles, time_text

    anim = animation.FuncAnimation(fig, update, frames=num_steps, blit=False, interval=1000 // fps)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".mp4":
        anim.save(str(out), writer="ffmpeg", fps=fps, dpi=dpi)
    else:
        anim.save(str(out), writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"Saved 3D flow animation to {out}")

    return anim


def plot_transport_3d_comparison(
    trajectories_dict: dict[str, np.ndarray],
    target_samples: np.ndarray,
    title: str = "3D OT-CFM Transport Trajectories & Regularization Comparison",
    figsize: tuple[float, float] = (16, 6),
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Multi-panel 3D comparison of particle flows across regularization conditions."""
    num_panels = len(trajectories_dict)
    fig = plt.figure(figsize=figsize, dpi=dpi)

    for idx, (label, traj) in enumerate(trajectories_dict.items()):
        ax = fig.add_subplot(1, num_panels, idx + 1, projection="3d")
        # Target samples
        ax.scatter(
            target_samples[:, 0],
            target_samples[:, 1],
            target_samples[:, 2],
            c="gray",
            alpha=0.1,
            s=8,
        )
        # Sample trajectories
        num_particles = min(120, traj.shape[1])
        cmap = plt.get_cmap("turbo")
        for p_i in range(num_particles):
            color = cmap(p_i / num_particles)
            ax.plot(
                traj[:, p_i, 0],
                traj[:, p_i, 1],
                traj[:, p_i, 2],
                color=color,
                alpha=0.45,
                linewidth=0.8,
            )
        # Final endpoints
        ax.scatter(
            traj[-1, :num_particles, 0],
            traj[-1, :num_particles, 1],
            traj[-1, :num_particles, 2],
            c="crimson",
            alpha=0.8,
            s=12,
        )
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_zlim(-3.5, 3.5)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$z$")
        ax.view_init(elev=25, azim=45)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved 3D comparison figure to {output_path}")

    return fig


def plot_transport_3d_pareto(
    summary_data: dict[str, Any],
    title: str = "3D OT-CFM Regularization & Integration Pareto Analysis",
    figsize: tuple[float, float] = (14, 5),
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot Wasserstein W2, SWD, and ODE step convergence across regularization strengths."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    raw_conditions = summary_data.get("conditions", [])
    if isinstance(raw_conditions, dict):
        conditions = list(raw_conditions.values())
    else:
        conditions = list(raw_conditions)

    # Sort conditions by beta ascending
    conditions.sort(key=lambda c: float(c.get("beta", 0.0)))

    betas = [float(c.get("beta", 0.0)) for c in conditions]
    w2_means = [
        float(c.get("empirical_w2_mean", c.get("w2_mean", 0.0))) for c in conditions
    ]
    w2_stds = [
        float(c.get("empirical_w2_std", c.get("w2_std", 0.0))) for c in conditions
    ]
    swd_means = [
        float(c.get("sliced_wasserstein_mean", c.get("swd_mean", 0.0))) for c in conditions
    ]
    swd_stds = [
        float(c.get("sliced_wasserstein_std", c.get("swd_std", 0.0))) for c in conditions
    ]

    # Panel 1: W2 vs beta
    ax1.errorbar(betas, w2_means, yerr=w2_stds, fmt="o-", color="navy", capsize=4, linewidth=1.8)
    ax1.set_xlabel(r"Regularization Strength $\beta$", fontsize=11)
    ax1.set_ylabel(r"Empirical $W_2$ Distance", fontsize=11)
    ax1.set_title(r"Transport Cost $W_2$ vs $\beta$", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: SWD vs beta
    ax2.errorbar(betas, swd_means, yerr=swd_stds, fmt="s-", color="crimson", capsize=4, linewidth=1.8)
    ax2.set_xlabel(r"Regularization Strength $\beta$", fontsize=11)
    ax2.set_ylabel(r"Sliced Wasserstein Distance (SWD)", fontsize=11)
    ax2.set_title(r"SWD Metric vs $\beta$", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Mode Coverage
    modes = [
        float(c.get("modes_covered_mean", c.get("mode_coverage_mean", 8.0))) for c in conditions
    ]
    ax3.bar(np.arange(len(betas)), modes, color="forestgreen", alpha=0.75, width=0.4)
    ax3.set_xticks(np.arange(len(betas)))
    ax3.set_xticklabels([f"$\\beta={b}$" for b in betas])
    ax3.set_ylim(0, 9)
    ax3.axhline(8.0, color="k", linestyle="--", alpha=0.7, label="All 8 Modes")
    ax3.set_title("Mode Coverage (8/8 Modes)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Covered Modes", fontsize=11)
    ax3.legend(frameon=True, fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved Pareto figure to {output_path}")

    return fig


def plot_hydrodynamic_3d_well(
    grid: Grid,
    potential: np.ndarray,
    density: np.ndarray,
    wavefunctions: np.ndarray,
    energies: np.ndarray,
    title: str = "3D Quantum Hydrodynamic State & Nodal Kinetic Energy",
    figsize: tuple[float, float] = (16, 5),
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Multi-panel 3D hydrodynamic visualization of potential, density, kinetic density, and energy levels."""
    nx, ny, nz = grid.points_per_dim
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # 1. 3D Potential Slices (2D middle cross section)
    ax1 = fig.add_subplot(1, 4, 1)
    v_slice = potential.reshape(nx, ny, nz)[:, :, nz // 2]
    im1 = ax1.imshow(
        v_slice.T,
        origin="lower",
        extent=[grid.bounds[0][0], grid.bounds[0][1], grid.bounds[1][0], grid.bounds[1][1]],
        cmap="viridis",
    )
    ax1.set_title("Potential $v(x, y, 0)$", fontweight="bold")
    ax1.set_xlabel("$x$ [Bohr]")
    ax1.set_ylabel("$y$ [Bohr]")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 2. Total Density n(x, y, 0)
    ax2 = fig.add_subplot(1, 4, 2)
    n_slice = density.reshape(nx, ny, nz)[:, :, nz // 2]
    im2 = ax2.imshow(
        n_slice.T,
        origin="lower",
        extent=[grid.bounds[0][0], grid.bounds[0][1], grid.bounds[1][0], grid.bounds[1][1]],
        cmap="inferno",
    )
    ax2.set_title("Electron Density $n(x, y, 0)$", fontweight="bold")
    ax2.set_xlabel("$x$ [Bohr]")
    ax2.set_ylabel("$y$ [Bohr]")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 3. Excited State Kinetic Energy Density tau(x, y, 0) = 1/2 |grad psi_1|^2
    ax3 = fig.add_subplot(1, 4, 3)
    psi1 = wavefunctions[:, min(1, wavefunctions.shape[1] - 1)].reshape(nx, ny, nz)
    grad_psi = np.gradient(psi1, *grid.spacings)
    tau = 0.5 * sum(g**2 for g in grad_psi)
    tau_slice = tau[:, :, nz // 2]
    im3 = ax3.imshow(
        tau_slice.T,
        origin="lower",
        extent=[grid.bounds[0][0], grid.bounds[0][1], grid.bounds[1][0], grid.bounds[1][1]],
        cmap="magma",
    )
    ax3.set_title(r"Nodal Kinetic Density $\tau_1$", fontweight="bold")
    ax3.set_xlabel("$x$ [Bohr]")
    ax3.set_ylabel("$y$ [Bohr]")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # 4. Energy Ladder
    ax4 = fig.add_subplot(1, 4, 4)
    num_e = min(6, len(energies))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i in range(num_e):
        e_i = float(energies[i])
        ax4.axhline(e_i, color=colors[i % len(colors)], linewidth=2.5, label=f"$\\epsilon_{i} = {e_i:.3f}$ Ha")
    ax4.set_xlim(-0.5, 0.5)
    ax4.set_xticks([])
    ax4.set_ylabel("Energy [Hartree]", fontsize=11)
    ax4.set_title("Eigenvalue Ladder", fontweight="bold")
    ax4.legend(loc="upper right", frameon=True, fontsize=9)
    ax4.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved hydrodynamic figure to {output_path}")

    return fig


def plot_multidim_inversion_comparison(
    grid: Grid,
    v_true: np.ndarray,
    v_recon: np.ndarray,
    n_exact: np.ndarray,
    n_relaxed: np.ndarray,
    title: str = "Multidimensional OF-DFT Potential & Ground-State Inversion",
    figsize: tuple[float, float] = (15, 5),
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Compare exact vs reconstructed potentials and relaxed ground state electron densities."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    shape = grid.points_per_dim

    if grid.dimension == 1:
        x = grid.coords_1d[0]
        ax1.plot(x, v_true, "k-", linewidth=2.0, label="True $v(x)$")
        ax1.plot(x, v_recon, "r--", linewidth=1.8, label="Reconstructed $v(x)$")
        ax1.set_xlabel("$x$ [Bohr]")
        ax1.set_ylabel("Potential [Hartree]")
        ax1.set_title("Potential Reconstruction", fontweight="bold")
        ax1.legend(frameon=True)
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2.plot(x, n_exact, "k-", linewidth=2.0, label=r"Exact $n^*(x)$")
        ax2.plot(x, n_relaxed, "b--", linewidth=1.8, label=r"Relaxed $n_{\mathrm{opt}}(x)$")
        ax2.fill_between(x, 0, n_exact, color="gray", alpha=0.2)
        ax2.set_xlabel("$x$ [Bohr]")
        ax2.set_ylabel("Density $n(x)$")
        ax2.set_title("Ground-State Density Matching", fontweight="bold")
        ax2.legend(frameon=True)
        ax2.grid(True, linestyle=":", alpha=0.6)

        residual = np.abs(n_exact - n_relaxed)
        ax3.plot(x, residual, "crimson", linewidth=1.8, label=r"Residual $|n^* - n_{\mathrm{opt}}|$")
        ax3.set_xlabel("$x$ [Bohr]")
        ax3.set_ylabel("Density Error")
        ax3.set_title("Variational Residual Error", fontweight="bold")
        ax3.legend(frameon=True)
        ax3.grid(True, linestyle=":", alpha=0.6)
    else:
        if grid.dimension == 2:
            slice_v_true = v_true.reshape(shape)
            slice_v_recon = v_recon.reshape(shape)
            slice_n_exact = n_exact.reshape(shape)
            slice_n_relaxed = n_relaxed.reshape(shape)
        else:
            slice_v_true = v_true.reshape(shape)[:, :, shape[2] // 2]
            slice_v_recon = v_recon.reshape(shape)[:, :, shape[2] // 2]
            slice_n_exact = n_exact.reshape(shape)[:, :, shape[2] // 2]
            slice_n_relaxed = n_relaxed.reshape(shape)[:, :, shape[2] // 2]

        extent = [grid.bounds[0][0], grid.bounds[0][1], grid.bounds[1][0], grid.bounds[1][1]]
        im1 = ax1.imshow(slice_v_true.T, origin="lower", extent=extent, cmap="viridis")
        ax1.set_title(r"Exact Potential $v(\mathbf{r})$", fontweight="bold")
        ax1.set_xlabel("$x$ [Bohr]")
        ax1.set_ylabel("$y$ [Bohr]")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        im2 = ax2.imshow(slice_v_recon.T, origin="lower", extent=extent, cmap="viridis")
        ax2.set_title(r"Reconstructed $v(\mathbf{r})$", fontweight="bold")
        ax2.set_xlabel("$x$ [Bohr]")
        ax2.set_ylabel("$y$ [Bohr]")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        im3 = ax3.imshow(
            np.abs(slice_n_exact - slice_n_relaxed).T,
            origin="lower",
            extent=extent,
            cmap="magma",
        )
        ax3.set_title(r"Density Residual $|n^* - n_{\mathrm{opt}}|$", fontweight="bold")
        ax3.set_xlabel("$x$ [Bohr]")
        ax3.set_ylabel("$y$ [Bohr]")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved multidimensional inversion figure to {output_path}")

    return fig


# ---------------------------------------------------------------------------
# 3D Grove Exhibition & WebGL Glowing Shader Exporters
# ---------------------------------------------------------------------------

GLOWING_ORBITAL_VERTEX_SHADER = """
varying vec3 vPosition;
varying vec3 vNormal;
varying vec2 vUv;
varying vec3 vWorldPosition;

void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    vPosition = position;
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
}
"""

RAYMARCHING_VOLUMETRIC_FRAGMENT_SHADER = """
precision highp float;

uniform float uTime;
uniform float uEnergy;
uniform vec3 uColorCore;
uniform vec3 uColorGlow;
uniform float uThreshold;
uniform sampler3D uDensityTexture;
uniform vec3 uCameraPos;

varying vec3 vPosition;
varying vec3 vNormal;
varying vec2 vUv;
varying vec3 vWorldPosition;

void main() {
    vec3 rayOrigin = vPosition;
    vec3 rayDir = normalize(vPosition - uCameraPos);
    
    // Raymarching parameters
    const int NUM_STEPS = 64;
    float stepSize = 1.732 / float(NUM_STEPS);
    vec3 currentPos = rayOrigin;
    
    vec4 accumulatedColor = vec4(0.0);
    
    for (int i = 0; i < NUM_STEPS; i++) {
        vec3 texCoord = currentPos * 0.5 + 0.5;
        if (texCoord.x >= 0.0 && texCoord.x <= 1.0 &&
            texCoord.y >= 0.0 && texCoord.y <= 1.0 &&
            texCoord.z >= 0.0 && texCoord.z <= 1.0) {
            
            float density = texture(uDensityTexture, texCoord).r;
            if (density > uThreshold * 0.1) {
                float intensity = smoothstep(uThreshold * 0.2, uThreshold, density);
                float pulse = 0.95 + 0.05 * sin(uTime * 3.0 + uEnergy * 2.0);
                vec3 stepEmission = mix(uColorGlow, uColorCore, intensity) * pulse;
                float stepAlpha = clamp(density * 0.05, 0.0, 1.0);
                
                // Front-to-back compositing
                accumulatedColor.rgb += (1.0 - accumulatedColor.a) * stepEmission * stepAlpha;
                accumulatedColor.a += (1.0 - accumulatedColor.a) * stepAlpha;
                
                if (accumulatedColor.a >= 0.98) break;
            }
        }
        currentPos += rayDir * stepSize;
    }
    
    // Fresnel rim accent
    vec3 viewDir = normalize(-vPosition);
    float fresnel = pow(1.0 - max(dot(vNormal, viewDir), 0.0), 3.0);
    accumulatedColor.rgb += uColorGlow * fresnel * 0.6;
    accumulatedColor.a = clamp(accumulatedColor.a + fresnel * 0.3, 0.0, 1.0);
    
    gl_FragColor = accumulatedColor;
}
"""

GLOWING_ORBITAL_FRAGMENT_SHADER = RAYMARCHING_VOLUMETRIC_FRAGMENT_SHADER


def export_grove_exhibition_manifest(
    output_dir: Path | str = "outputs/grove_exhibition",
    grid_points: int = 24,
    num_orbitals: int = 4,
) -> dict[str, Any]:
    """Generate comprehensive 3D exhibition room artifacts, plaques, and shaders for weichseltree.com/grove."""
    import json

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    grid = Grid.create(dimension=3, lower=-3.0, upper=3.0, points=grid_points)
    coords = grid.coordinates
    # Generate an illustrative 3D multi-well molecular-like potential
    r1 = coords - np.array([-1.0, 0.0, 0.0])
    r2 = coords - np.array([1.0, 0.0, 0.0])
    v_3d = (
        -12.0 * np.exp(-np.sum(r1**2, axis=-1) / 1.5)
        - 12.0 * np.exp(-np.sum(r2**2, axis=-1) / 1.5)
        + 0.15 * np.sum(coords**2, axis=-1)
    )

    sol = solve_multidim_schroedinger(potential=v_3d, grid=grid, num_orbitals=num_orbitals)
    orbital_energies = sol["orbital_energies"].tolist()
    densities_3d = sol["density"].reshape((grid_points, grid_points, grid_points)).tolist()

    state_labels = ["Ground State", "1st Excited State", "2nd Excited State", "3rd Excited State"]
    orbitals_data = []
    orbitals_3d_stack = []
    for k in range(num_orbitals):
        psi_k = sol["wavefunctions"][:, k]
        dens_k = (psi_k**2).reshape((grid_points, grid_points, grid_points))
        orbitals_3d_stack.append(dens_k)
        label = state_labels[k] if k < len(state_labels) else f"Excited State #{k}"
        orbitals_data.append(
            {
                "orbital_index": k,
                "state_label": label,
                "energy_hartree": float(orbital_energies[k]),
                "harmonic_frequency_rad_s": float(abs(orbital_energies[k])),
                "max_density": float(np.max(dens_k)),
                "core_color": ["#00f0ff", "#ff007f", "#7000ff", "#00ff66"][k % 4],
                "glow_color": ["#0088ff", "#ff44aa", "#aa44ff", "#44ff88"][k % 4],
                "nodal_surfaces": k,
                "plaque": {
                    "title": f"Stationary Eigenstate \u03c8_{k}(\u0302r)",
                    "energy": f"{float(orbital_energies[k]):.4f} Ha",
                    "nodal_topology": (
                        "0 nodal planes (single continuous probability volume)"
                        if k == 0
                        else f"{k} zero-crossing nodal plane(s) separating alternating phase lobes"
                    ),
                    "harmonic_phase": f"Time-evolution phase pulsation frequency \u03c9 = {abs(float(orbital_energies[k])):.4f} rad/s [exp(-i E t / \u210f)]",
                    "kinetic_localization": "Local kinetic energy density \u03c4(r) = 1/2|\u2207\u03c8|^2 peaks in thin luminous bands hugging zero-crossing boundaries",
                    "physics": (
                        "Ground state: no nodal planes, single coherent lobe, lowest possible kinetic energy."
                        if k == 0
                        else f"Kinetic energy concentrates at zero-crossing nodes. State #{k + 1} with {k} nodal plane(s) sits strictly above lower states."
                    ),
                    "physical_meaning": (
                        "The fundamental ground state minimizes spatial curvature |\u2207\u03c8|^2, spreading symmetrically across both potential wells."
                        if k == 0
                        else f"Nodal boundaries force steep spatial gradients \u2207\u03c8, driving up the kinetic energy integral T_k = 1/2 \u222b |\u2207\u03c8_k|^2 dr and pushing state #{k + 1} upward in the quantum energy ladder."
                    ),
                    "how_to_read": "The intense glowing core denotes peak electron density |\u03c8|^2; the outer Fresnel rim accent captures quantum tunneling decay into the confining barrier.",
                },
            }
        )

    manifest = {
        "title": "QuantumFlow Exhibition — Physics Turned Into Standing 3D Objects",
        "subtitle": "From Single-Particle Wavefunctions to Many-Body Density Inversion and Optimal Transport Flows",
        "grove_route": "weichseltree.com/grove/quantumflow",
        "theme": "The Hohenberg-Kohn Reduction: Can an electron density alone know everything about the quantum matter that created it?",
        "curator_statement": (
            "Quantum mechanics is conventionally presented as a bewildering hierarchy of wavefunctions "
            "\u03c8_1(r), ..., \u03c8_N(r), each oscillating through zero and demanding immense computational "
            "budgets to avoid nodal catastrophes. This spatial exhibition walks the visitor through the "
            "complete physics of density functional theory (DFT), verifying how a single scalar electron "
            "cloud n(r) captures total quantum reality without losing a single physical observable. "
            "From standing physical orbitals in the Orangery to falsifiable convexity bowls, generative "
            "optimal transport flow cubes, quantum hydrodynamic pressure fields, and exact variational "
            "potential inversions, every exhibit carries a measurable quantity painted directly onto its "
            "surfaces for visitors to inspect and falsify."
        ),
        "narrative_arc": [
            "1. The Quantum Energy Ladder: Why nodes cost kinetic energy and how total density encloses charge without zero-crossings.",
            "2. Optimal Transport & Generative Flows: Why minimal-action straight geodesics beat curvature constraints.",
            "3. Quantum Hydrodynamics: Bohm potential, quantum pressure, and kinetic localization.",
            "4. The Inversion Chamber: Exact potential reconstruction through input-convex neural functionals.",
            "5. Holographic Raymarching: Dynamic phase oscillation and volumetric isosurface topology.",
        ],
        "rooms": [
            {
                "id": "room_1_orangery",
                "gallery_number": 1,
                "title": "Gallery 1: The Orangery & The Quantum Energy Ladder",
                "theme": "Single-Particle Wavefunctions vs. Total Electron Density",
                "curator_hook": (
                    "Every orbital shape in this room floats at an elevation strictly proportional to its eigenvalue energy. "
                    "Walk the gallery and observe why state #2 sits above state #1: kinetic energy is concentrated directly in the nodal gaps."
                ),
                "concept_explanation": (
                    "In Schrödinger wave mechanics, boundary confinement forces standing waves. Higher quantum states "
                    "develop additional zero-crossing nodal surfaces where \u03c8(r) = 0. Because the wavefunction must "
                    "transition from positive to negative across a finite distance, spatial gradients |\u2207\u03c8| steepen "
                    "drastically at the nodes, driving up the kinetic energy integral T = 1/2 \u222b |\u2207\u03c8|^2 dr. "
                    "Meanwhile, the total electron cloud n(r) = \u2211_i |\u03c8_i(r)|^2 sums all occupied states into a strictly "
                    "positive, smooth density field without nodes."
                ),
                "mathematical_thesis": (
                    "Nodal count monotonically forces kinetic energy elevation: T_k = 1/2 \u222b |\u2207\u03c8_k|^2 dr, "
                    "whereas the total density n(r) is strictly positive everywhere."
                ),
                "falsifiable_criterion": (
                    "Count the nodal gaps on each orbital model and check the vertex luminance: the brightest regions "
                    "must align with the zero-crossing boundaries, and the energy heights must match the eigenvalue ladder."
                ),
                "visitor_reading_guide": (
                    "1. Gold lobes indicate positive phase (\u03c8 > 0); cyan lobes indicate negative phase (\u03c8 < 0). "
                    "2. The physical gap separating lobes is the nodal surface where \u03c8 = 0. "
                    "3. Look for the luminous yellow glow hugging the node: this is the local kinetic energy density \u03c4(r). "
                    "4. Compare the multi-lobed orbitals against the nested grey density shells: notice that the density has no lobes or gaps."
                ),
                "boundary_limitations": (
                    "These are exact solutions for non-interacting electrons in a representative double-well potential, "
                    "not empirical multi-electron molecular spectra. Vertical height scale is calibrated to 0.9048 m per Hartree."
                ),
                "description": (
                    "Physical standing eigenstates lit by local kinetic energy density tau = 1/2|grad psi|^2. "
                    "Nested containment shells enclosing 50%, 80%, and 95% electron probability, and "
                    "falsifiable convexity bowls carrying green/red chord ribbons."
                ),
                "artifacts": [
                    {
                        "path": "figures/figure_1d_orbitals.svg",
                        "kind": "figure",
                        "title": "1D Numerov Eigenstates & Nodal Energy Progression",
                        "description": "Exact 1D Schrödinger bound states showing potential well v(x), energy levels \u03b5_k, and wavefunctions \u03c8_k(x) with k=0, 1, 2, 3 nodes.",
                        "physical_insight": "Each additional node forces an extra oscillation within the confining well, increasing the second derivative \u03c8''(x) and driving up the orbital energy.",
                        "how_to_read": "Observe how the ground state (k=0) has zero zero-crossings, while each excited state adds exactly one nodal point.",
                        "boundary_caveat": "Solved on a 1D grid with 128 points under Dirichlet boundary conditions.",
                    },
                    {
                        "path": "figures/figure_1d_reconstruction.png",
                        "kind": "figure",
                        "title": "Potential Inversion & Density Fidelity in 1D",
                        "description": "Direct benchmark comparison of true external potential v_true(x) against reconstructed potential v_pred(x) derived from ground-state density.",
                        "physical_insight": "Demonstrates that slight deviations in external potential propagate directly into visible shifts in density profile and energy eigenvalues.",
                        "how_to_read": "Dashed lines represent predicted fields; solid curves indicate exact ground-truth.",
                        "boundary_caveat": "Synthetic perturbation tested against baseline Numerov solution.",
                    },
                ],
                "models": [
                    "results/models/orbital-0.glb",
                    "results/models/orbital-1.glb",
                    "results/models/density-shells.glb",
                    "results/models/potential-relief.glb",
                    "results/models/convex-bowl.glb",
                    "results/models/unconstrained-bowl.glb",
                ],
            },
            {
                "id": "room_2_flow_cube",
                "gallery_number": 2,
                "title": "Gallery 2: The 3D Flow Cube (Optimal Transport & Flow Matching)",
                "theme": "Generative Velocity Fields & The Cost of Artificial Constraints",
                "curator_hook": (
                    "We tested a hypothesis that imposing an isotropic Hessian penalty would regularize generative flow matching. "
                    "The empirical outcome was conclusive: unconstrained straight geodesics won on every metric."
                ),
                "concept_explanation": (
                    "Optimal Transport Conditional Flow Matching (OT-CFM) constructs a time-dependent vector field "
                    "v_t(x) = \u2207\u03a6(t, x) that pushes a simple prior (standard 3D Gaussian) into a multimodal target "
                    "distribution (8 Gaussian modes located at the corners of a cube (\u00b12, \u00b12, \u00b12)). "
                    "Monge-Kantorovich optimal transport couples source and target points along straight, constant-velocity "
                    "geodesic trajectories x_t = (1-t) x_0 + t x_1. When we introduced a regularizer "
                    "\u03b2 ||\u2207^2 \u03a6 - \u03bb I||^2 to force uniform spatial curvature, the added penalty fought against "
                    "the natural multi-branching flow, increasing Wasserstein distance W_2 and sliced Wasserstein distance "
                    "at every non-zero strength."
                ),
                "mathematical_thesis": (
                    "Straight OT geodesics minimize kinetic action S = \u222b_0^1 ||v_t(x_t)||^2 dt. "
                    "Enforcing Hessian isotropy \u03b2 > 0 increases both 2-Wasserstein error and transport variance without improving mode coverage."
                ),
                "falsifiable_criterion": (
                    "Examine the 3D particle trajectories: the baseline \u03b2 = 0 produces perfectly straight, direct stream "
                    "paths into all 8 modes (W_2 = 0.7741), whereas \u03b2 = 2.0 introduces path deflection and dispersion."
                ),
                "visitor_reading_guide": (
                    "1. In the animation, watch particles originate at the center spherical cloud (t=0) and branch outwards into the 8 cube vertices (t=1). "
                    "2. Compare the side-by-side trajectories of \u03b2=0 vs \u03b2=2: notice that \u03b2=0 maintains tighter target mode clusters. "
                    "3. In the Pareto chart, note that increasing \u03b2 shifts the performance curve upwards and to the right (worse performance)."
                ),
                "boundary_limitations": (
                    "Flow potential \u03a6(t, x) is defined up to an arbitrary time-dependent additive gauge constant c(t), "
                    "meaning height coordinates between independently trained runs are not directly comparable."
                ),
                "description": (
                    "3D Optimal Transport Conditional Flow Matching (OT-CFM) transporting a spherical Gaussian "
                    "distribution into 8 cube Gaussian modes at (\u00b12, \u00b12, \u00b12). Straight geodesic paths achieve "
                    "W2 = 0.7741 with full 8/8 mode coverage, outperforming constrained curvature penalties."
                ),
                "artifacts": [
                    {
                        "path": "videos/transport_3d_cube_flow.gif",
                        "kind": "animation",
                        "title": "3D OT-CFM Cube Branching Dynamics",
                        "description": "Time-resolved integration of 400 test particles flowing from standard Gaussian prior into 8 distinct Gaussian cube clusters.",
                        "physical_insight": "Particles follow straight optimal-transport trajectories that smoothly bifurcate into 8 symmetric 3D octants without trajectory crossing.",
                        "how_to_read": "Color coding reflects particle velocity magnitude along the dynamic trajectory.",
                        "boundary_caveat": "Integrated using Runge-Kutta 4th order (RK4) with 30 time steps.",
                    },
                    {
                        "path": "videos/transport_flow.gif",
                        "kind": "animation",
                        "title": "2D 8-Gaussian Ring Transport Flow",
                        "description": "2D generative flow matching benchmark transporting a central normal distribution into an 8-mode ring manifold.",
                        "physical_insight": "Demonstrates radial symmetric transport and optimal mass allocation across circular modes.",
                        "how_to_read": "Track individual particle paths as they radially disperse into distinct angular modes.",
                        "boundary_caveat": "2D baseline model evaluated over 350 particles.",
                    },
                    {
                        "path": "figures/figure_3d_transport_comparison.png",
                        "kind": "figure",
                        "title": "Trajectory Geometry: Baseline \u03b2=0 vs Regularized \u03b2=2.0",
                        "description": "Comparative 3D trajectory plot showing particle stream paths under unconstrained geodesics vs isotropic Hessian penalty.",
                        "physical_insight": "Visual evidence of how the Hessian penalty distorts linear geodesics into curved paths, degrading cluster sharpness at destination modes.",
                        "how_to_read": "Blue streams show unconstrained straight flow; orange streams show curvature-penalized paths.",
                        "boundary_caveat": "Evaluated on 400 test particles with identical random seed initialization.",
                    },
                    {
                        "path": "figures/figure_3d_transport_pareto.png",
                        "kind": "figure",
                        "title": "Pareto Frontier: Transport Error vs Regularization Strength",
                        "description": "Quantitative evaluation of 2-Wasserstein distance W_2, Sliced Wasserstein Distance (SWD), and Mode Coverage across \u03b2 in [0.0, 2.0].",
                        "physical_insight": "The unconstrained baseline \u03b2 = 0 sits at the strict Pareto optimum. Every non-zero \u03b2 strictly increases transport error while mode coverage remains constant at 8/8.",
                        "how_to_read": "Lower y-axis values indicate superior transport accuracy. Note the monotone increase in W_2 as \u03b2 increases.",
                        "boundary_caveat": "Averaged over multiple evaluation seeds with standard error bands shown.",
                    },
                ],
            },
            {
                "id": "room_3_hydrodynamics",
                "gallery_number": 3,
                "title": "Gallery 3: Quantum Hydrodynamics & Nodal Kinetic Field",
                "theme": "Madelung Fluid Representation & Quantum Pressure",
                "curator_hook": (
                    "What prevents an electron cloud from collapsing into the bottom of a potential well? "
                    "The answer is quantum pressure — an emergent hydrodynamic force arising from spatial curvature of the density."
                ),
                "concept_explanation": (
                    "By substituting the polar decomposition \u03c8(r) = \u221an(r) exp(i S(r)/\u210f) into the time-dependent "
                    "Schrödinger equation, Erwin Madelung showed that quantum mechanics maps exactly onto the Euler fluid equations. "
                    "The fluid velocity is v = 1/m \u2207S, and the Euler momentum equation acquires an extra term: the Bohm "
                    "quantum potential Q(r) = -\u210f^2/(2m) (\u2207^2 \u221an)/\u221an. The corresponding quantum kinetic energy splits "
                    "into the von Weizsäcker term T_W = 1/8 \u222b |\u2207n|^2/n dr (representing density compression) and a "
                    "Pauli/nodal kinetic term (representing wavefunction oscillations). Where wavefunctions vanish at nodes, "
                    "\u2207n / n steepens, generating intense localized quantum pressure that counteracts external electrostatic attraction."
                ),
                "mathematical_thesis": (
                    "The total kinetic energy decomposes into hydrodynamic quantum pressure and nodal velocity components: "
                    "T = T_W[n] + T_Pauli = 1/8 \u222b |\u2207n|^2 / n dr + 1/2 \u222b n |\u2207S|^2 dr."
                ),
                "falsifiable_criterion": (
                    "Inspect the 3D hydrodynamic cross-sections: verify that the quantum pressure Q(r) peaks where the "
                    "density exhibits maximum second-derivative curvature, establishing mechanical equilibrium against \u2207v."
                ),
                "visitor_reading_guide": (
                    "1. Panel 1 shows the confining 3D molecular potential v(r). "
                    "2. Panel 2 shows the resulting ground-state electron density n(r). "
                    "3. Panel 3 shows the kinetic energy density field \u03c4(r). "
                    "4. Panel 4 displays the Bohm quantum potential Q(r), highlighting regions where quantum pressure acts as an effective repulsive force."
                ),
                "boundary_limitations": (
                    "Computed for a stationary ground-state system where S(r) = const (zero hydrodynamic current), "
                    "isolating the pure quantum pressure contribution."
                ),
                "description": (
                    "3D Schrödinger fluid mechanics and Bohm quantum pressure TW = |grad n|^2 / (8n). "
                    "Visualization of kinetic energy density concentrating in razor-thin luminous bands "
                    "at nodal surfaces where wavefunctions vanish."
                ),
                "artifacts": [
                    {
                        "path": "figures/figure_3d_hydrodynamic_well.png",
                        "kind": "figure",
                        "title": "3D Quantum Hydrodynamic Equilibrium",
                        "description": "Four-panel 2D/3D slice analysis illustrating confining potential v(r), total electron density n(r), kinetic energy density \u03c4(r), and Bohm quantum potential Q(r).",
                        "physical_insight": "Direct visualization of the balance between electrostatic potential confinement and Bohm quantum pressure.",
                        "how_to_read": "Compare the contours of Q(r) in the outer barrier with the decay of n(r): sharp decay in density corresponds to steep positive quantum potential barriers.",
                        "boundary_caveat": "Rendered on a 24x24x24 grid with cubic interpolation for smooth vector gradient fields.",
                    },
                    {
                        "path": "figures/figure_3d_hydrodynamic_well.svg",
                        "kind": "figure",
                        "title": "Vector Graphics: 3D Hydrodynamic Cross-Sections",
                        "description": "High-resolution vector rendering of quantum hydrodynamic balance fields.",
                        "physical_insight": "Clean vector visualization suitable for detailed architectural wall presentation.",
                        "how_to_read": "Vector contour lines represent constant probability density isosurfaces.",
                        "boundary_caveat": "Vector export matching the raster PNG dataset.",
                    },
                ],
            },
            {
                "id": "room_4_inversion",
                "gallery_number": 4,
                "title": "Gallery 4: The Inversion Chamber (Orbital-Free DFT & Relaxation)",
                "theme": "The Hohenberg-Kohn Inversion: Retrieving v(r) from n(r) Alone",
                "curator_hook": (
                    "Hand the machine an electron cloud and it hands back the potential that shaped it — "
                    "with no orbitals, no wavefunctions, and zero chance of getting stuck in false local minima."
                ),
                "concept_explanation": (
                    "The first Hohenberg-Kohn theorem (1964) establishes that the ground-state electron density n(r) "
                    "uniquely determines the external potential v(r) up to an additive constant. In Orbital-Free DFT, "
                    "ground-state density is found by minimizing the total energy functional E[n] = T_s[n] + \u222b v(r) n(r) dr "
                    "subject to \u222b n(r) dr = N. At the variational minimum, the Euler-Lagrange equation requires "
                    "\u03bc = \u03b4T_s[n] / \u03b4n(r) + v(r). If the kinetic functional T_s[n] is parameterized by an Input-Convex "
                    "Neural Network (ICNN), where all feedforward weights are strictly non-negative (W \u2265 0 via Softplus), "
                    "the energy functional E[n] is guaranteed to be strictly convex. Variational density relaxation is thus "
                    "mathematically guaranteed to converge to the unique global ground state."
                ),
                "mathematical_thesis": (
                    "Exact functional inversion: v(r) = \u03bc - \u03b4T_s[n] / \u03b4n(r). "
                    "Convexity of T_s[n] guarantees that variational optimization min_n E[n] is devoid of spurious local minima."
                ),
                "falsifiable_criterion": (
                    "Inspect the Convexity Bowl models in Gallery 1: the ICNN bowl contains exactly 0 violating red chords "
                    "across all random affine slices, whereas the unconstrained network produces 93+ violating chords."
                ),
                "visitor_reading_guide": (
                    "1. In the density relaxation video, observe how the density field starts as a uniform distribution and smoothly flows into the double-well ground state. "
                    "2. In the inversion comparison plot, check the overlay between v_true and v_recon: the median reconstruction error across held-out systems proves the density retains complete potential information."
                ),
                "boundary_limitations": (
                    "Reconstruction quality depends on the expressive capacity of the neural functional architecture. "
                    "Inversion determines v(r) up to an arbitrary additive chemical potential constant \u03bc."
                ),
                "description": (
                    "Variational Euler-Lagrange relaxation mu = delta Ts/delta n + v using Input-Convex Neural "
                    "Networks (ICNN). Hand the machine a cloud and reconstruct the external well that shaped it."
                ),
                "artifacts": [
                    {
                        "path": "figures/figure_2d_system.svg",
                        "kind": "figure",
                        "title": "2D Multi-Well Quantum Molecular System",
                        "description": "2D benchmark system showing potential well landscape, ground-state density, individual orbitals, and energy spectrum.",
                        "physical_insight": "Demonstrates how multi-center potentials create localized bound states and bonding/antibonding density distributions in 2D.",
                        "how_to_read": "Top-left: potential surface; Top-right: ground-state density; Bottom row: orbital eigenfunctions.",
                        "boundary_caveat": "Solved on a 32x32 2D spatial grid.",
                    },
                    {
                        "path": "figures/figure_3d_inversion_comparison.png",
                        "kind": "figure",
                        "title": "3D Multidimensional Potential Inversion",
                        "description": "Direct visual comparison between 3D ground-truth potential v_true and functional-inverted potential v_recon along key spatial cuts.",
                        "physical_insight": "Validates that the functional derivative \u03b4T_s / \u03b4n accurately reconstructs the 3D potential well geometry from the relaxed density alone.",
                        "how_to_read": "Blue curves represent exact target fields; dashed red curves represent neural functional reconstruction.",
                        "boundary_caveat": "Evaluated across 3D molecular potential slice.",
                    },
                    {
                        "path": "videos/density_relaxation.gif",
                        "kind": "animation",
                        "title": "Variational Density Relaxation Dynamics",
                        "description": "Step-by-step optimization trajectory of density logits relaxing under Adam optimizer on the convex energy surface E[n].",
                        "physical_insight": "Illustrates the smooth, monotonic decrease in total variational energy and convergence to the exact ground-state density profile without oscillations or trapping.",
                        "how_to_read": "Top curve tracks density profile evolution; bottom curve shows strict monotonic descent of variational energy.",
                        "boundary_caveat": "Optimized over 40 gradient steps with particle number conservation enforced via softplus-softmax normalization.",
                    },
                ],
            },
            {
                "id": "room_5_hologram",
                "gallery_number": 5,
                "title": "Gallery 5: Holographic 3D Volumetric Raymarching Chamber",
                "theme": "Interactive Volumetric Emission & Quantum Phase Dynamics",
                "curator_hook": (
                    "Step inside the volumetric raymarching chamber. Each glowing eigenstate pulsates at its own quantum harmonic "
                    "frequency \u03c9_k = E_k / \u210f, turning abstract energy eigenvalues into physical light rhythms."
                ),
                "concept_explanation": (
                    "Standard 3D mesh approximations extract polygonal isosurfaces, discarding internal density variations. "
                    "This interactive WebGL chamber implements real-time volumetric raymarching directly through 3D floating-point "
                    "texture grids. As the viewer's ray traverses the 3D volume, it computes front-to-back optical absorption "
                    "and emission based on local density n(x, y, z). Each orbital eigenstate is modulated by a harmonic phase "
                    "pulse factor 1.0 + 0.05 sin(\u03c9_k t + 2 E_k), mapping quantum phase dynamics into visible luminescence. "
                    "Fresnel rim lighting highlights exponential tunneling decay at the boundary of the classically forbidden barrier."
                ),
                "mathematical_thesis": (
                    "Volumetric radiative transport: I(x, \u03c9) = \u222b_0^s \u03c3(x_t) C(x_t) exp(-\u222b_0^t \u03c3(x_u) du) dt, "
                    "with quantum phase modulation \u03c8_k(r, t) = \u03c8_k(r) exp(-i E_k t / \u210f)."
                ),
                "falsifiable_criterion": (
                    "Verify that lower energy states (e.g. state #1 at -8.4742 Ha) oscillate at distinct visible frequencies "
                    "compared to higher states (state #4 at -5.1170 Ha), and check that Fresnel highlights accurately trace the exponential boundary decay."
                ),
                "visitor_reading_guide": (
                    "1. Orbit the camera to observe the 3D volumetric depth: the inner core represents maximum probability density, while the outer halo shows exponential barrier tunneling. "
                    "2. Observe the rhythmic pulsing: each state pulses at a unique tempo determined by its exact eigenvalue energy. "
                    "3. Look for the rim accent along the silhouette edge: this reflects the gradient of the potential barrier."
                ),
                "boundary_limitations": (
                    "Raymarched with 64 front-to-back integration steps on a 24x24x24 grid. Phase pulse frequency is visually scaled for human ocular perception."
                ),
                "description": (
                    "Interactive WebGL volumetric raymarching chamber rendering 3D electron densities and "
                    "orbital eigenstates with dynamic phase-pulsing energy eigenvalues and Fresnel rim lighting."
                ),
                "artifacts": [
                    {
                        "path": "density_3d.npy",
                        "kind": "volumetric_data",
                        "title": "3D Floating-Point Volumetric Electron Density Matrix",
                        "description": "Continuous 3D scalar density field array n(x, y, z) for WebGL 3D texture sampler upload.",
                        "physical_insight": "Encodes continuous spatial electron distribution n(r) for volumetric optical raymarching.",
                        "how_to_read": "Loaded into sampler3D texture unit; mapped to absorption and emission transfer functions.",
                        "boundary_caveat": "Float32 array at grid resolution.",
                    },
                    {
                        "path": "orbitals_3d.npy",
                        "kind": "volumetric_data",
                        "title": "3D Volumetric Wavefunction Eigenstate Stack",
                        "description": "Multi-channel 3D array of squared orbital densities |\u03c8_k(r)|^2.",
                        "physical_insight": "Provides independent volumetric channels for real-time eigenstate raymarching and dynamic phase pulsing.",
                        "how_to_read": "Individual channels correspond to sequential energy eigenstates.",
                        "boundary_caveat": "Stack of 4 float32 3D volumes.",
                    },
                ],
                "shader_type": "VolumetricRaymarching",
                "shaders": {
                    "vertex": GLOWING_ORBITAL_VERTEX_SHADER,
                    "fragment": RAYMARCHING_VOLUMETRIC_FRAGMENT_SHADER,
                },
                "grid": {
                    "dimension": 3,
                    "resolution": [grid_points, grid_points, grid_points],
                    "bounds": [[-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0]],
                },
                "orbitals": orbitals_data,
            },
        ],
    }

    manifest_file = out_path / "grove_exhibition_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Export volumetric density and orbital matrices as binary numpy arrays for WebGL texture upload
    np.save(out_path / "density_3d.npy", np.asarray(densities_3d, dtype=np.float32))
    np.save(out_path / "orbitals_3d.npy", np.asarray(orbitals_3d_stack, dtype=np.float32))

    print(f"Exported Grove exhibition manifest and 3D assets to {out_path.resolve()}")
    return manifest


