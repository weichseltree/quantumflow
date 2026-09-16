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


# ---------------------------------------------------------------------------
# 3D Grove Exhibition & WebGL Glowing Shader Exporters
# ---------------------------------------------------------------------------

GLOWING_ORBITAL_VERTEX_SHADER = """
varying vec3 vPosition;
varying vec3 vNormal;
varying vec2 vUv;

void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    vPosition = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
"""

GLOWING_ORBITAL_FRAGMENT_SHADER = """
precision highp float;

uniform float uTime;
uniform float uEnergy;
uniform vec3 uColorCore;
uniform vec3 uColorGlow;
uniform float uThreshold;
uniform sampler3D uDensityTexture;

varying vec3 vPosition;
varying vec3 vNormal;
varying vec2 vUv;

void main() {
    // Coordinate normalized to [0, 1]^3 for 3D volumetric sampling
    vec3 texCoord = vPosition * 0.5 + 0.5;
    float density = texture(uDensityTexture, texCoord).r;

    // Volumetric glow calculation
    float intensity = smoothstep(uThreshold * 0.2, uThreshold, density);
    float glow = pow(density / (uThreshold + 1e-4), 1.8);

    // Fresnel rim lighting for holographic exhibition aesthetic
    vec3 viewDir = normalize(-vPosition);
    float fresnel = pow(1.0 - max(dot(vNormal, viewDir), 0.0), 3.0);

    // Dynamic pulsating emission based on orbital energy eigenvalue
    float pulse = 0.9 + 0.1 * sin(uTime * 2.5 + uEnergy * 3.0);
    vec3 color = mix(uColorGlow, uColorCore, intensity) * glow * pulse;
    color += uColorGlow * fresnel * 0.8;

    float alpha = clamp(glow * 0.85 + fresnel * 0.5, 0.0, 1.0);
    gl_FragColor = vec4(color, alpha);
}
"""


def export_grove_exhibition_manifest(
    output_dir: Path | str = "outputs/grove_exhibition",
    grid_points: int = 16,
    num_orbitals: int = 3,
) -> dict[str, Any]:
    """Generate 3D exhibition room artifacts and shaders for weichseltree.com/grove."""
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

    orbitals_data = []
    for k in range(num_orbitals):
        psi_k = sol["wavefunctions"][:, k]
        dens_k = (psi_k**2).reshape((grid_points, grid_points, grid_points))
        orbitals_data.append(
            {
                "orbital_index": k,
                "energy_hartree": float(orbital_energies[k]),
                "max_density": float(np.max(dens_k)),
                "core_color": ["#00f0ff", "#ff007f", "#7000ff"][k % 3],
                "glow_color": ["#0088ff", "#ff44aa", "#aa44ff"][k % 3],
            }
        )

    manifest = {
        "title": "QuantumFlow Exhibition — From 1D Numerov to 3D Neural Orbitals",
        "grove_route": "weichseltree.com/grove/quantumflow",
        "rooms": [
            {
                "id": "room_1_origins",
                "title": "Gallery 1: 1D DFT & Exact Numerov Shooting",
                "description": (
                    "Historical 1D kinetic models (Snyder 2012, Alghadeer 2021) "
                    "and node-matching Numerov solutions."
                ),
                "artifacts": [
                    "figures/figure_1d_orbitals.svg",
                    "figures/figure_1d_reconstruction.png",
                ],
            },
            {
                "id": "room_2_transport",
                "title": "Gallery 2: Optimal Transport & Flow Matching",
                "description": (
                    "Continuous Normalizing Flows regularized with Isotropic Hessian penalties."
                ),
                "artifacts": ["videos/transport_flow.gif"],
            },
            {
                "id": "room_3_multidim_solvers",
                "title": "Gallery 3: Multi-Dimensional Convex Potentials & ICNN",
                "description": (
                    "2D and 3D kinetic energy functionals via Input Convex Neural Networks "
                    "with Euler potential reconstruction."
                ),
                "artifacts": ["figures/figure_2d_system.svg", "videos/density_relaxation.gif"],
            },
            {
                "id": "room_4_hologram",
                "title": "Gallery 4: Glowing 3D Quantum Shader Chamber",
                "description": (
                    "Interactive raymarched 3D electron densities and orbital eigenstates "
                    "hovering in spatial Grove rooms."
                ),
                "shader_type": "VolumetricGlow",
                "shaders": {
                    "vertex": GLOWING_ORBITAL_VERTEX_SHADER,
                    "fragment": GLOWING_ORBITAL_FRAGMENT_SHADER,
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

    # Export volumetric density matrix as binary numpy/json for WebGL texture upload
    np.save(out_path / "density_3d.npy", np.asarray(densities_3d, dtype=np.float32))

    print(f"Exported Grove exhibition manifest and 3D assets to {out_path.resolve()}")
    return manifest
