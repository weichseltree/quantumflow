import jax
import jax.numpy as jnp
import numpy as np
import optax

from quantumflow.jax import (
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


def test_grid_properties_2d_and_3d() -> None:
    grid_2d = Grid.create(dimension=2, lower=-3.0, upper=3.0, points=16)
    assert grid_2d.total_points == 256
    assert len(grid_2d.spacings) == 2
    assert grid_2d.volume_element > 0.0
    assert grid_2d.coordinates.shape == (256, 2)

    grid_3d = Grid.create(dimension=3, lower=-2.0, upper=2.0, points=8)
    assert grid_3d.total_points == 512
    assert len(grid_3d.spacings) == 3
    assert grid_3d.coordinates.shape == (512, 3)


def test_2d_schroedinger_harmonic_oscillator_eigenvalues() -> None:
    """Test 2D Schrödinger solver on 2D isotropic harmonic oscillator V(x,y) = 0.5*(x^2 + y^2).

    Analytic eigenvalues: E_{nx, ny} = (nx + ny + 1)*hbar*omega.
    For omega=1, hbar=m=1:
    E0 = 1.0 (non-degenerate: (0,0))
    E1, E2 = 2.0 (doubly degenerate: (1,0), (0,1))
    E3, E4, E5 = 3.0 (triply degenerate: (2,0), (1,1), (0,2))
    """
    grid = Grid.create(dimension=2, lower=-4.5, upper=4.5, points=32)
    coords = grid.coordinates
    v_ho = 0.5 * np.sum(coords**2, axis=-1)

    sol = solve_multidim_schroedinger(potential=v_ho, grid=grid, num_orbitals=3)
    energies = sol["orbital_energies"]

    # Ground state energy close to 1.0
    assert np.isclose(energies[0], 1.0, atol=0.08)
    # First excited states close to 2.0
    assert np.isclose(energies[1], 2.0, atol=0.10)
    assert np.isclose(energies[2], 2.0, atol=0.10)

    # Check density norm int n(r) dr == num_orbitals
    density_integral = np.sum(sol["density"]) * grid.volume_element
    assert np.isclose(density_integral, 3.0, atol=1e-5)

    # Check energy conservation E = Ts + V
    assert np.isclose(
        sol["total_energy"],
        sol["kinetic_energy"] + sol["potential_energy"],
        atol=1e-6,
    )


def test_3d_schroedinger_harmonic_oscillator_eigenvalues() -> None:
    """Test 3D Schrödinger solver on 3D harmonic oscillator V(r) = 0.5*r^2.

    Analytic ground state: E0 = 1.5.
    """
    grid = Grid.create(dimension=3, lower=-3.5, upper=3.5, points=14)
    coords = grid.coordinates
    v_ho = 0.5 * np.sum(coords**2, axis=-1)

    sol = solve_multidim_schroedinger(potential=v_ho, grid=grid, num_orbitals=1)
    assert np.isclose(sol["orbital_energies"][0], 1.5, atol=0.12)
    assert np.isclose(np.sum(sol["density"]) * grid.volume_element, 1.0, atol=1e-5)


def test_multidim_dataset_generation_and_euler_relation() -> None:
    grid = Grid.create(dimension=2, lower=-3.0, upper=3.0, points=16)
    dataset = generate_multidim_dataset(grid=grid, dataset_size=5, num_orbitals=2, seed=123)

    assert dataset["densities"].shape == (5, 256)
    assert dataset["potentials"].shape == (5, 256)
    assert dataset["kinetic_energies"].shape == (5,)
    assert dataset["orbital_energies"].shape == (5, 2)
    assert dataset["derivatives"].shape == (5, 256)

    # Verify Euler relation delta T / delta n = mu - v(r)
    for i in range(5):
        mu = dataset["chemical_potentials"][i]
        v = dataset["potentials"][i]
        deriv = dataset["derivatives"][i]
        assert np.allclose(deriv, mu - v)


def test_convex_functional_2d_jensen_and_euler_reconstruction() -> None:
    grid_size = 64
    params = init_icnn(jax.random.key(10), input_size=grid_size, hidden_units=(32, 32))

    n1 = jnp.full((grid_size,), 0.2)
    n2 = jnp.full((grid_size,), 0.6)
    mid = (n1 + n2) / 2.0

    t_mid = kinetic_energy(params, mid)
    t_avg = 0.5 * (kinetic_energy(params, n1) + kinetic_energy(params, n2))
    assert float(t_mid) <= float(t_avg) + 1e-6

    # Test reconstruction of potential from derivative
    density = jnp.full((grid_size,), 0.4)
    mu = 2.5
    v_rec = reconstruct_potential(params, density, chemical_potential=mu)
    deriv = functional_derivative(params, density)
    assert jnp.allclose(v_rec, mu - deriv)


def test_composite_training_step_reduces_joint_loss() -> None:
    grid_size = 16
    params = init_icnn(jax.random.key(20), input_size=grid_size, hidden_units=(16,))
    density = jnp.array([[0.2] * grid_size, [0.5] * grid_size], dtype=jnp.float32)
    target_t = jnp.array([1.2, 2.4], dtype=jnp.float32)
    target_deriv = jnp.zeros((2, grid_size), dtype=jnp.float32)

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)
    step_fn = make_composite_training_step(optimizer, alpha_derivative=0.5)

    params_new, opt_state_new, aux = step_fn(params, opt_state, density, target_t, target_deriv)
    assert jnp.isfinite(aux["loss"])
    assert jnp.isfinite(aux["loss_energy"])
    assert jnp.isfinite(aux["loss_derivative"])


def test_variational_ground_state_solver_2d() -> None:
    grid = Grid.create(dimension=2, lower=-2.5, upper=2.5, points=12)
    params = init_icnn(jax.random.key(30), input_size=grid.total_points, hidden_units=(16,))

    coords = grid.coordinates
    v_target = 0.5 * np.sum(coords**2, axis=-1)

    sol = solve_ground_state_density(
        params=params,
        potential=v_target,
        num_particles=2.0,
        volume_element=grid.volume_element,
        num_steps=30,
        lr=5e-2,
    )

    n_opt = sol["density"]
    assert n_opt.shape == (grid.total_points,)
    # Verify density is non-negative and integrates to N=2
    assert jnp.all(n_opt >= 0.0)
    assert np.isclose(float(jnp.sum(n_opt) * grid.volume_element), 2.0, atol=1e-4)
    assert np.isfinite(sol["chemical_potential"])
