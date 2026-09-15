import jax
import jax.numpy as jnp
import numpy as np

from quantumflow.ot_cfm import (
    compute_eigenvalue_spread,
    empirical_wasserstein_distance,
    evaluate_potential,
    exact_ot_coupling,
    init_potential_network,
    integrate_ode,
    sample_8gaussians,
    sliced_wasserstein_distance,
    velocity,
)
from scripts.run_transport import train_and_eval


def test_init_and_evaluate_potential() -> None:
    key = jax.random.key(0)
    params = init_potential_network(key, in_dim=2, hidden_dims=(32, 32))
    t = jnp.array(0.5)
    x = jnp.array([1.0, -1.0])
    out = evaluate_potential(params, t, x)
    assert out.shape == ()
    v = velocity(params, t, x)
    assert v.shape == (2,)


def test_sample_8gaussians_and_ot_coupling() -> None:
    key = jax.random.key(1)
    samples = sample_8gaussians(key, batch_size=100)
    assert samples.shape == (100, 2)

    x0 = np.random.randn(64, 2)
    x1 = np.asarray(samples[:64])
    x0_paired, x1_paired = exact_ot_coupling(x0, x1)
    assert x0_paired.shape == (64, 2)
    assert x1_paired.shape == (64, 2)


def test_ode_integration_and_metrics() -> None:
    key = jax.random.key(2)
    params = init_potential_network(key, in_dim=2, hidden_dims=(16, 16))
    x0 = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    final_x, traj = integrate_ode(params, x0, num_steps=5)
    assert final_x.shape == (2, 2)
    assert len(traj) == 6

    spread = compute_eigenvalue_spread(params, x0, t=0.5)
    assert spread >= 0.0

    swd = sliced_wasserstein_distance(final_x, x0, key=key)
    assert swd >= 0.0


def test_empirical_wasserstein_distance_is_true_assignment_metric() -> None:
    x = jnp.array([[0.0, 0.0], [2.0, 0.0]])
    y = jnp.array([[1.0, 0.0], [3.0, 0.0]])
    assert empirical_wasserstein_distance(x, y) == 1.0


def test_empirical_wasserstein_distance_rejects_unequal_samples() -> None:
    with np.testing.assert_raises(ValueError):
        empirical_wasserstein_distance(jnp.zeros((2, 2)), jnp.zeros((3, 2)))


def test_training_evaluation_is_reproducible(tmp_path) -> None:
    first = train_and_eval(beta=0.0, num_steps=1, batch_size=4, seed=17, eval_samples=8)
    second = train_and_eval(beta=0.0, num_steps=1, batch_size=4, seed=17, eval_samples=8)

    deterministic_fields = (
        "final_loss",
        "final_loss_cfm",
        "final_loss_iso",
        "sliced_wasserstein_distance",
        "empirical_wasserstein_distance",
        "mean_eigenvalue_spread",
        "trajectory_eigenvalue_spreads",
        "pareto_step_sliced_wasserstein_distance",
        "ode_function_evaluations",
        "evaluation_seed",
        "projection_seed",
    )
    for field in deterministic_fields:
        np.testing.assert_equal(first[field], second[field])
