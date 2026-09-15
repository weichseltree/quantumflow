import jax.numpy as jnp
import numpy as np

from quantumflow.transport import hessian, isotropic_hessian_penalty, trace_free_hessian


def test_hessian_is_computed_per_sample_for_a_batched_potential() -> None:
    def potential(points: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * jnp.sum(points**2, axis=1)

    points = jnp.array([[1.0, 2.0], [-3.0, 4.0]])

    np.testing.assert_allclose(
        hessian(potential, points),
        np.broadcast_to(np.eye(2), (2, 2, 2)),
    )


def test_isotropic_hessian_has_no_penalty() -> None:
    def potential(points: jnp.ndarray) -> jnp.ndarray:
        return 3.0 * jnp.sum(points**2, axis=1)

    points = jnp.array([[1.0, 2.0], [-3.0, 4.0]])

    np.testing.assert_allclose(isotropic_hessian_penalty(potential, points), 0.0)
    np.testing.assert_allclose(trace_free_hessian(potential, points), 0.0)


def test_penalty_measures_only_the_anisotropic_part() -> None:
    def potential(points: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * jnp.sum(points**2 * jnp.array([2.0, 4.0]), axis=1)

    points = jnp.array([[1.0, 0.0], [0.0, 1.0]])

    # H = diag(2, 4), so H - tr(H)I/2 = diag(-1, 1).
    assert isotropic_hessian_penalty(potential, points) == 2.0
