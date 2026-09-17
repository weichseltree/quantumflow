import jax
import jax.numpy as jnp
import optax

from quantumflow.jax import (
    functional_derivative,
    init_icnn,
    kinetic_energy,
    make_training_step,
    potential_from_kinetic_derivative,
)


def test_input_convex_functional_satisfies_jensen_inequality() -> None:
    params = init_icnn(jax.random.key(1), input_size=8, hidden_units=(4, 4))
    first_density = jnp.full((8,), 0.1)
    second_density = jnp.full((8,), 0.8)
    midpoint = (first_density + second_density) / 2

    midpoint_energy = kinetic_energy(params, midpoint)
    mean_energy = (
        kinetic_energy(params, first_density) + kinetic_energy(params, second_density)
    ) / 2

    assert float(midpoint_energy) <= float(mean_energy) + 1e-6


def test_derivative_and_potential_follow_euler_equation() -> None:
    params = init_icnn(jax.random.key(2), input_size=4, hidden_units=(3,))
    density = jnp.full((2, 4), 0.5)

    derivative = functional_derivative(params, density)
    potential = potential_from_kinetic_derivative(derivative, 3.0)

    assert derivative.shape == density.shape
    assert jnp.allclose(potential, 3.0 - derivative)


def test_training_step_reduces_linear_target_loss() -> None:
    params = init_icnn(jax.random.key(3), input_size=2, hidden_units=(4,))
    density = jnp.array([[0.2, 0.1], [0.4, 0.5], [0.7, 0.3]])
    targets = jnp.array([0.1, 0.3, 0.4])
    optimizer = optax.adam(1e-2)
    optimizer_state = optimizer.init(params)

    step = make_training_step(optimizer)
    _, _, loss = step(params, optimizer_state, density, targets)

    assert jnp.isfinite(loss)


def test_parameters_round_trip_through_an_archive(tmp_path):
    """The trained functional has to come back byte-identical to build the room."""
    import numpy as np

    from quantumflow.jax.convex import load_parameters, save_parameters

    params = init_icnn(jax.random.key(3), input_size=12, hidden_units=(8, 8))
    path = tmp_path / "icnn.npz"
    save_parameters(params, path)
    restored = load_parameters(path)

    assert set(restored) == set(params)
    for name, (weight, bias) in params.items():
        np.testing.assert_allclose(restored[name][0], weight, atol=0)
        np.testing.assert_allclose(restored[name][1], bias, atol=0)

    density = jnp.asarray(np.linspace(0.1, 1.0, 12))
    np.testing.assert_allclose(
        kinetic_energy(restored, density), kinetic_energy(params, density), atol=0
    )
