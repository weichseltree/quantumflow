"""Differentiable utilities for isotropic-Hessian transport constraints."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

Potential = Callable[[jax.Array], jax.Array]


def hessian(potential: Potential, points: jax.Array) -> jax.Array:
    """Return one Hessian matrix per point for a scalar potential.

    ``potential`` must return one scalar value per row of ``points`` (either
    shape ``(batch,)`` or ``(batch, 1)``).
    """
    points = jnp.asarray(points)
    if points.ndim != 2:
        raise ValueError("points must have shape (batch, dimension)")

    def scalar_potential(point: jax.Array) -> jax.Array:
        return jnp.reshape(potential(point[jnp.newaxis, :]), ())

    sample_hessian = jax.jacfwd(jax.grad(scalar_potential))
    return jax.vmap(sample_hessian)(points)


def trace_free_hessian(potential: Potential, points: jax.Array) -> jax.Array:
    """Return the trace-free part of each potential Hessian."""
    matrices = hessian(potential, points)
    dimension = matrices.shape[-1]
    trace = jnp.trace(matrices, axis1=-2, axis2=-1)[..., jnp.newaxis, jnp.newaxis]
    identity = jnp.eye(dimension, dtype=matrices.dtype)
    return matrices - trace * identity / dimension


def isotropic_hessian_penalty(potential: Potential, points: jax.Array) -> jax.Array:
    """Return the mean squared Frobenius norm of the trace-free Hessian.

    The penalty is zero exactly when every Hessian is a scalar multiple of the
    identity, which is the isotropic-Hessian constraint
    ``nabla^2 Phi = lambda(x) I``.
    """
    trace_free = trace_free_hessian(potential, points)
    return jnp.mean(jnp.sum(jnp.square(trace_free), axis=(-2, -1)))
