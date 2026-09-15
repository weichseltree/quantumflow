"""Differentiable utilities for isotropic-Hessian transport constraints."""

from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf

Potential = Callable[[tf.Tensor], tf.Tensor]


def hessian(potential: Potential, points: tf.Tensor) -> tf.Tensor:
    """Return one Hessian matrix per point for a scalar potential.

    ``potential`` must return one scalar value per row of ``points`` (either
    shape ``(batch,)`` or ``(batch, 1)``). The full Jacobians are reduced to
    their per-sample blocks, avoiding cross-sample derivatives when the
    potential is implemented as a batched Keras model.
    """
    points = tf.convert_to_tensor(points)
    if points.shape.rank != 2:
        raise ValueError("points must have shape (batch, dimension)")

    with tf.GradientTape() as outer_tape:
        outer_tape.watch(points)
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(points)
            values = tf.reshape(potential(points), [-1])
        gradients = inner_tape.jacobian(values, points)
        gradients = tf.transpose(tf.linalg.diag_part(tf.transpose(gradients, [2, 0, 1])))
    jacobian = outer_tape.jacobian(gradients, points)
    return tf.transpose(tf.linalg.diag_part(tf.transpose(jacobian, [1, 3, 0, 2])), [2, 0, 1])


def trace_free_hessian(potential: Potential, points: tf.Tensor) -> tf.Tensor:
    """Return the trace-free part of each potential Hessian."""
    matrices = hessian(potential, points)
    dimension = tf.cast(tf.shape(matrices)[-1], matrices.dtype)
    trace = tf.linalg.trace(matrices)[..., tf.newaxis, tf.newaxis]
    identity = tf.eye(tf.shape(matrices)[-1], dtype=matrices.dtype)
    return matrices - trace * identity / dimension


def isotropic_hessian_penalty(potential: Potential, points: tf.Tensor) -> tf.Tensor:
    """Return the mean squared Frobenius norm of the trace-free Hessian.

    The penalty is zero exactly when every Hessian is a scalar multiple of the
    identity, which is the isotropic-Hessian constraint
    ``nabla^2 Phi = lambda(x) I``.
    """
    trace_free = trace_free_hessian(potential, points)
    return tf.reduce_mean(tf.reduce_sum(tf.square(trace_free), axis=(-2, -1)))
