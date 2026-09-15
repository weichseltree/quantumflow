"""Optimal Transport Conditional Flow Matching with Isotropic-Hessian Regularization."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from quantumflow.transport import isotropic_hessian_penalty


class ModelParams(NamedTuple):
    weights: list[jax.Array]
    biases: list[jax.Array]


def save_model_params(params: ModelParams, path: Path) -> None:
    """Save model parameters in a portable, non-executable NumPy archive."""
    arrays = {"layer_count": np.array(len(params.weights), dtype=np.int64)}
    for index, (weight, bias) in enumerate(zip(params.weights, params.biases, strict=True)):
        arrays[f"weight_{index}"] = np.asarray(weight)
        arrays[f"bias_{index}"] = np.asarray(bias)
    np.savez_compressed(path, **arrays)


def load_model_params(path: Path) -> ModelParams:
    """Load model parameters written by :func:`save_model_params`."""
    with np.load(path, allow_pickle=False) as archive:
        layer_count = int(archive["layer_count"])
        weights = [jnp.asarray(archive[f"weight_{index}"]) for index in range(layer_count)]
        biases = [jnp.asarray(archive[f"bias_{index}"]) for index in range(layer_count)]
    return ModelParams(weights=weights, biases=biases)


def init_potential_network(
    key: jax.Array,
    in_dim: int = 2,
    hidden_dims: tuple[int, ...] = (128, 128, 128),
) -> ModelParams:
    """Initialize an MLP scalar potential network Phi(t, x) -> scalar."""
    layer_dims = (in_dim + 1, *hidden_dims, 1)
    keys = jax.random.split(key, len(layer_dims) - 1)
    weights = []
    biases = []
    layer_pairs = zip(layer_dims[:-1], layer_dims[1:], strict=True)
    for k, (d_in, d_out) in zip(keys, layer_pairs, strict=True):
        limit = jnp.sqrt(6.0 / (d_in + d_out))
        w = jax.random.uniform(k, (d_in, d_out), minval=-limit, maxval=limit)
        b = jnp.zeros((d_out,))
        weights.append(w)
        biases.append(b)
    return ModelParams(weights=weights, biases=biases)


def evaluate_potential(params: ModelParams, t: jax.Array, x: jax.Array) -> jax.Array:
    """Evaluate scalar potential Phi(t, x) for a single point x in R^d and scalar t.

    Input: t shape (), x shape (d,).
    Output: scalar shape ().
    """
    t_expanded = jnp.atleast_1d(t)
    x_flat = jnp.reshape(x, (-1,))
    h = jnp.concatenate([t_expanded, x_flat], axis=0)

    num_layers = len(params.weights)
    for i in range(num_layers - 1):
        h = h @ params.weights[i] + params.biases[i]
        h = jax.nn.silu(h)
    out = h @ params.weights[-1] + params.biases[-1]
    return jnp.squeeze(out)


def velocity(params: ModelParams, t: jax.Array, x: jax.Array) -> jax.Array:
    """Compute curl-free velocity v(t, x) = grad_x Phi(t, x) for a single point x."""
    return jax.grad(lambda x_: evaluate_potential(params, t, x_))(x)


def batched_velocity(params: ModelParams, t: jax.Array, x: jax.Array) -> jax.Array:
    """Compute velocity for batched inputs x (N, d) at time t (N,) or scalar t."""
    if t.ndim == 0:
        return jax.vmap(lambda x_: velocity(params, t, x_))(x)
    return jax.vmap(lambda t_, x_: velocity(params, t_, x_))(t, x)


def sample_8gaussians(
    key: jax.Array,
    batch_size: int,
    radius: float = 2.0,
    std: float = 0.1,
) -> jax.Array:
    """Sample from an 8-Gaussian mixture on a ring of given radius."""
    key_comp, key_noise = jax.random.split(key)
    angles = jnp.linspace(0, 2 * jnp.pi, 9)[:-1]
    centers = jnp.stack([radius * jnp.cos(angles), radius * jnp.sin(angles)], axis=1)

    components = jax.random.choice(key_comp, 8, shape=(batch_size,))
    chosen_centers = centers[components]
    noise = jax.random.normal(key_noise, shape=(batch_size, 2)) * std
    return chosen_centers + noise


def exact_ot_coupling(x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pair minibatches using exact 2D quadratic Wasserstein optimal transport."""
    cost_matrix = cdist(x0, x1, metric="sqeuclidean")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return x0[row_ind], x1[col_ind]


def cfm_loss_step(
    params: ModelParams,
    t: jax.Array,
    x0: jax.Array,
    x1: jax.Array,
    beta: float = 0.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute OT-CFM loss + beta * L_iso for a batch of pairs."""
    t_col = t[:, jnp.newaxis]
    x_t = (1.0 - t_col) * x0 + t_col * x1
    u_t = x1 - x0

    v_t = batched_velocity(params, t, x_t)
    loss_cfm = jnp.mean(jnp.sum(jnp.square(v_t - u_t), axis=-1))

    if beta > 0.0:
        # Sample isotropic penalty at a mean time slice or across random times
        def sample_penalty(t_s, x_s):
            def pot_t(x_batch):
                return jax.vmap(lambda x_row: evaluate_potential(params, t_s, x_row))(x_batch)

            return isotropic_hessian_penalty(pot_t, x_s[jnp.newaxis, :])

        penalties = jax.vmap(sample_penalty)(t, x_t)
        loss_iso = jnp.mean(penalties)
    else:
        loss_iso = jnp.array(0.0, dtype=loss_cfm.dtype)

    total_loss = loss_cfm + beta * loss_iso
    metrics = {
        "loss": total_loss,
        "loss_cfm": loss_cfm,
        "loss_iso": loss_iso,
    }
    return total_loss, metrics


def rk4_step(params: ModelParams, t: float, x: jax.Array, dt: float) -> jax.Array:
    """Single Runge-Kutta 4 integration step for dx/dt = v(t, x)."""
    k1 = batched_velocity(params, jnp.array(t), x)
    k2 = batched_velocity(params, jnp.array(t + 0.5 * dt), x + 0.5 * dt * k1)
    k3 = batched_velocity(params, jnp.array(t + 0.5 * dt), x + 0.5 * dt * k2)
    k4 = batched_velocity(params, jnp.array(t + dt), x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_ode(
    params: ModelParams,
    x0: jax.Array,
    num_steps: int = 50,
) -> tuple[jax.Array, list[jax.Array]]:
    """Integrate velocity field from t=0 to t=1 using RK4.

    Returns:
        (final_samples, trajectory_list)
    """
    dt = 1.0 / num_steps
    x = x0
    trajectory = [x]
    for step in range(num_steps):
        t = step * dt
        x = rk4_step(params, t, x, dt)
        trajectory.append(x)
    return x, trajectory


def compute_mode_metrics(
    samples: jax.Array | np.ndarray,
    radius: float = 2.0,
    threshold: float = 0.5,
) -> dict[str, float | int | list[int]]:
    """Compute mode coverage and statistics for 8-Gaussian ring target."""
    samples_np = np.asarray(samples)
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    dists = cdist(samples_np, centers)
    closest_mode = np.argmin(dists, axis=1)
    min_dist = np.min(dists, axis=1)

    valid_mask = min_dist <= threshold
    valid_modes = closest_mode[valid_mask]

    counts = np.bincount(valid_modes, minlength=8).tolist()
    total_assigned = int(sum(counts))
    modes_covered = int(sum(1 for c in counts if c > 0))
    missing_modes = int(8 - modes_covered)

    if total_assigned > 0:
        probs = np.array(counts, dtype=float) / total_assigned
        non_zero_p = probs[probs > 0]
        entropy = -float(np.sum(non_zero_p * np.log2(non_zero_p)))
        norm_entropy = float(entropy / 3.0)  # log2(8) = 3.0
    else:
        norm_entropy = 0.0

    return {
        "modes_covered": modes_covered,
        "missing_modes": missing_modes,
        "mode_counts": counts,
        "mode_entropy": norm_entropy,
        "unassigned_samples": int(len(samples_np) - total_assigned),
    }


def compute_eigenvalue_spread(
    params: ModelParams,
    points: jax.Array,
    t: float,
) -> dict[str, jax.Array]:
    """Compute eigenvalue spread |lambda_1 - lambda_2| / (|lambda_1| + |lambda_2| + eps).

    Returns a dict with 'mean' and 'max' spread.
    """
    def point_hessian(p):
        return jax.hessian(lambda p_: evaluate_potential(params, jnp.array(t), p_))(p)

    hessians = jax.vmap(point_hessian)(points)  # (N, 2, 2)
    eigvals = jax.vmap(jnp.linalg.eigvalsh)(hessians)  # (N, 2)
    l1 = eigvals[:, 0]
    l2 = eigvals[:, 1]
    spread = jnp.abs(l1 - l2) / (jnp.abs(l1) + jnp.abs(l2) + 1e-6)
    return {
        "mean": jnp.mean(spread),
        "max": jnp.max(spread),
    }


def sliced_wasserstein_distance(
    x: jax.Array,
    y: jax.Array,
    key: jax.Array,
    num_projections: int = 128,
) -> jax.Array:
    """Compute Sliced 2-Wasserstein distance between empirical distributions in 2D."""
    theta = jax.random.uniform(key, (num_projections,), minval=0.0, maxval=2.0 * jnp.pi)
    projections = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)  # (P, 2)

    proj_x = x @ projections.T  # (N, P)
    proj_y = y @ projections.T  # (M, P)

    sorted_x = jnp.sort(proj_x, axis=0)
    sorted_y = jnp.sort(proj_y, axis=0)

    if x.shape[0] != y.shape[0]:
        raise ValueError("sliced Wasserstein currently requires equal sample counts")
    sq_diff = jnp.mean(jnp.square(sorted_x - sorted_y), axis=0)
    return jnp.sqrt(jnp.mean(sq_diff))


def empirical_wasserstein_distance(
    x: jax.Array,
    y: jax.Array,
) -> float:
    """Return the empirical quadratic Wasserstein distance for equal-size samples.

    The assignment solver computes the exact discrete optimum for the uniform
    empirical measures. This is a true W2 distance, unlike the sliced metric.
    """
    x_np = np.asarray(x)
    y_np = np.asarray(y)
    if x_np.ndim != 2 or y_np.ndim != 2 or x_np.shape[1] != y_np.shape[1]:
        raise ValueError("x and y must have shape (samples, dimension)")
    if x_np.shape[0] != y_np.shape[0]:
        raise ValueError("empirical Wasserstein distance requires equal sample counts")
    cost_matrix = cdist(x_np, y_np, metric="sqeuclidean")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return float(np.sqrt(np.mean(cost_matrix[row_ind, col_ind])))
