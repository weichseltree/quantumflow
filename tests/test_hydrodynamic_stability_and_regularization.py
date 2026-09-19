"""Tests for quantum fluid hydrodynamic stability and 3D features."""

from __future__ import annotations

import jax
import numpy as np

from quantumflow.jax.convex import (
    init_icnn,
    kinetic_energy,
    multispecies_kinetic_energy,
    quantum_potential,
    regularized_kinetic_energy,
    solve_ground_state_density_stabilized,
    spin_resolved_kinetic_energy,
    von_weizsaecker_kinetic_energy,
)
from quantumflow.multidim import (
    Grid,
    generate_bosonic_dataset,
    generate_spin_polarized_dataset,
    quantum_potential_multidim,
    von_weizsaecker_multidim,
)
from quantumflow.ot_cfm import (
    compute_eigenvalue_spread,
    compute_mode_metrics_3d,
    init_potential_network,
    sample_cube_gaussians_3d,
    sliced_wasserstein_distance,
)


def test_von_weizsaecker_1d_gaussian_ground_state():
    """Verify T_W[n] on analytical ground state psi_0 = (1/pi)^(1/4) exp(-x^2/2).

    Theoretical kinetic energy is exactly T_W = 0.25 Hartree.
    """
    n_pts = 401
    x = np.linspace(-6.0, 6.0, n_pts)
    h = x[1] - x[0]
    psi_0 = (1.0 / np.pi) ** 0.25 * np.exp(-0.5 * x**2)
    density = psi_0**2

    # Verify integral of density = 1
    assert np.isclose(np.sum(density) * h, 1.0, atol=1e-5)

    tw = float(von_weizsaecker_kinetic_energy(density, spacing=h))
    assert np.isclose(tw, 0.25, atol=1e-3)


def test_quantum_potential_1d():
    """Verify Q[n](x) = -1/2 (psi''(x)/psi(x)) on harmonic ground state.

    For psi_0(x) = exp(-x^2/2), psi''(x) = (x^2 - 1) psi_0(x), so Q(x) = 1/2 - x^2/2.
    In potential v(x) = x^2/2, Euler sum Q(x) + v(x) = 1/2 (exact constant chemical potential mu).
    """
    n_pts = 201
    x = np.linspace(-4.0, 4.0, n_pts)
    h = x[1] - x[0]
    density = (1.0 / np.pi) ** 0.5 * np.exp(-(x**2))

    q_pot = np.asarray(quantum_potential(density, spacing=h))
    v_ext = 0.5 * x**2
    euler_sum = q_pot + v_ext

    # In the core region (-2 to 2) where density is non-negligible, sum should be constant 0.5
    core_mask = np.abs(x) < 2.0
    assert np.allclose(euler_sum[core_mask], 0.5, atol=2e-2)


def test_regularized_kinetic_energy_lower_bound():
    """Verify T_s[n] = T_W[n] + T_corr[n] satisfies T_s[n] >= T_W[n] when T_corr >= 0."""
    key = jax.random.key(42)
    n_pts = 50
    params = init_icnn(key, input_size=n_pts, hidden_units=(32, 32))

    density = np.abs(np.random.default_rng(42).standard_normal((5, n_pts))) + 0.1
    tw = np.asarray(von_weizsaecker_kinetic_energy(density, spacing=0.1))
    t_corr = np.asarray(kinetic_energy(params, density))
    t_reg = np.asarray(regularized_kinetic_energy(params, density, spacing=0.1, c_w=1.0))

    assert np.all(t_corr >= -1e-6)
    assert np.all(t_reg >= tw - 1e-6)


def test_spin_resolved_kinetic_energy():
    """Verify Oliver-Perdew relation T_s[n_up, n_down] = 1/2 T_s[2 n_up] + 1/2 T_s[2 n_down]."""
    key = jax.random.key(123)
    n_pts = 40
    params = init_icnn(key, input_size=n_pts, hidden_units=(32, 32))

    n_up = np.abs(np.random.default_rng(0).standard_normal(n_pts)) + 0.1
    n_down = np.abs(np.random.default_rng(1).standard_normal(n_pts)) + 0.1

    t_spin = float(spin_resolved_kinetic_energy(params, n_up, n_down, spacing=0.1))
    t_up_2 = float(regularized_kinetic_energy(params, 2.0 * n_up, spacing=0.1))
    t_down_2 = float(regularized_kinetic_energy(params, 2.0 * n_down, spacing=0.1))

    assert np.isclose(t_spin, 0.5 * (t_up_2 + t_down_2), atol=1e-5)


def test_multispecies_mass_weighting():
    """Verify mass scaling in multispecies kinetic energy functional."""
    key = jax.random.key(456)
    n_pts = 30
    params = init_icnn(key, input_size=n_pts, hidden_units=(16, 16))

    n_1 = np.ones(n_pts) * 0.5
    n_2 = np.ones(n_pts) * 0.8
    masses = [1.0, 2.0]

    t_multi = float(multispecies_kinetic_energy(params, [n_1, n_2], masses, spacing=0.1))
    t_1 = float(regularized_kinetic_energy(params, n_1, spacing=0.1))
    t_2 = float(regularized_kinetic_energy(params, n_2, spacing=0.1))

    expected = 0.5 * t_1 + 0.25 * t_2
    assert np.isclose(t_multi, expected, atol=1e-5)


def test_stabilized_ground_state_solver():
    """Verify solve_ground_state_density_stabilized converges and returns normalized density."""
    key = jax.random.key(789)
    n_pts = 50
    params = init_icnn(key, input_size=n_pts, hidden_units=(32, 32))
    v = np.linspace(-2.0, 2.0, n_pts) ** 2

    res = solve_ground_state_density_stabilized(
        params=params,
        potential=v,
        num_particles=2.0,
        volume_element=0.1,
        c_w=1.0,
        num_steps=100,
    )

    density = np.asarray(res["density"])
    assert np.all(density >= 0.0)
    assert np.isclose(np.sum(density) * 0.1, 2.0, atol=1e-4)
    assert np.isfinite(res["kinetic_energy"])
    assert np.isfinite(res["chemical_potential"])


def test_multidim_von_weizsaecker_and_quantum_potential_2d():
    """Verify 2D multidimensional von Weizsaecker functional and quantum potential on grid."""
    grid = Grid.create(dimension=2, lower=-3.0, upper=3.0, points=20)
    coords = grid.coordinates
    r_sq = np.sum(coords**2, axis=-1)
    # 2D Gaussian ground state: psi_0(r) = (1/pi)^(1/2) exp(-r^2/2)
    psi_0 = (1.0 / np.pi) ** 0.5 * np.exp(-0.5 * r_sq)
    density = psi_0**2

    tw_2d = float(von_weizsaecker_multidim(density, grid))
    # In 2D isotropic harmonic oscillator, E_kin = 2 * (1/4) = 0.5 Hartree
    assert np.isclose(tw_2d, 0.5, atol=2e-2)

    q_pot_2d = quantum_potential_multidim(density, grid)
    assert q_pot_2d.shape == (grid.total_points,)
    assert np.all(np.isfinite(q_pot_2d))


def test_bosonic_and_spin_polarized_datasets():
    """Verify generation of bosonic and spin-polarized multi-dimensional datasets."""
    grid = Grid.create(dimension=1, lower=-3.0, upper=3.0, points=30)

    boson_data = generate_bosonic_dataset(grid, dataset_size=5, num_particles=4.0, seed=42)
    assert boson_data["densities"].shape == (5, 30)
    # Verify normalization
    for i in range(5):
        assert np.isclose(np.sum(boson_data["densities"][i]) * grid.volume_element, 4.0, atol=1e-5)

    spin_data = generate_spin_polarized_dataset(grid, dataset_size=5, num_up=2, num_down=1, seed=42)
    assert spin_data["densities_up"].shape == (5, 30)
    assert spin_data["densities_down"].shape == (5, 30)
    assert spin_data["densities_total"].shape == (5, 30)
    for i in range(5):
        assert np.isclose(
            np.sum(spin_data["densities_up"][i]) * grid.volume_element, 2.0, atol=1e-5
        )
        assert np.isclose(
            np.sum(spin_data["densities_down"][i]) * grid.volume_element, 1.0, atol=1e-5
        )


def test_3d_cube_gaussians_and_metrics():
    """Verify 3D cube Gaussian sampling, mode metrics, and d-dimensional sliced Wasserstein."""
    key = jax.random.key(42)
    samples_3d = sample_cube_gaussians_3d(key, batch_size=500, scale=2.0, std=0.05)
    assert samples_3d.shape == (500, 3)

    metrics_3d = compute_mode_metrics_3d(samples_3d, scale=2.0, threshold=0.5)
    assert metrics_3d["modes_covered"] == 8
    assert metrics_3d["missing_modes"] == 0
    assert metrics_3d["mode_entropy"] > 0.95

    # 3D Sliced Wasserstein
    target_3d = sample_cube_gaussians_3d(jax.random.key(43), batch_size=500, scale=2.0, std=0.05)
    swd = float(
        sliced_wasserstein_distance(
            samples_3d, target_3d, key=jax.random.key(44), num_projections=64
        )
    )
    assert np.isfinite(swd)
    assert swd < 0.8
    # Exact match for identical distributions
    swd_identical = float(
        sliced_wasserstein_distance(
            samples_3d, samples_3d, key=jax.random.key(44), num_projections=64
        )
    )
    assert np.isclose(swd_identical, 0.0, atol=1e-5)


def test_3d_eigenvalue_spread():
    """Verify 3D eigenvalue spread calculation for potential network."""
    key = jax.random.key(101)
    params = init_potential_network(key, in_dim=3, hidden_dims=(32, 32))
    pts_3d = jax.random.normal(jax.random.key(102), shape=(10, 3))

    spread_dict = compute_eigenvalue_spread(params, pts_3d, t=0.5)
    assert "mean" in spread_dict
    assert "max" in spread_dict
    assert 0.0 <= float(spread_dict["mean"]) <= 1.0
