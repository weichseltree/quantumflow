"""Convex neural kinetic-energy functionals."""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf


def InputConvexKineticEnergyFunctional(
    run_dir,
    dataset,
    hidden_units: Sequence[int] = (128, 128),
    l2_regularisation: float = 0.0,
    **kwargs,
):
    """Build an input-convex neural network for the discretized kinetic energy.

    The hidden-to-hidden and output weights are constrained to be non-negative,
    while Softplus is convex and non-decreasing.  The resulting scalar kinetic
    energy is therefore convex with respect to its discretized density input.
    """
    if not hidden_units or any(units <= 0 for units in hidden_units):
        raise ValueError("hidden_units must contain one or more positive integers")

    regularizer = tf.keras.regularizers.l2(l2_regularisation) if l2_regularisation > 0 else None
    density = tf.keras.layers.Input(shape=dataset.density.shape[1:], name="density")
    flattened_density = tf.keras.layers.Flatten(name="flatten_density")(density)

    value = tf.keras.layers.Dense(
        hidden_units[0],
        activation="softplus",
        kernel_regularizer=regularizer,
        name="convex_hidden_0",
    )(flattened_density)
    for index, units in enumerate(hidden_units[1:], start=1):
        convex_term = tf.keras.layers.Dense(
            units,
            use_bias=False,
            kernel_constraint=tf.keras.constraints.NonNeg(),
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.05),
            kernel_regularizer=regularizer,
            name=f"convex_hidden_{index}",
        )(value)
        affine_term = tf.keras.layers.Dense(
            units,
            use_bias=True,
            kernel_regularizer=regularizer,
            name=f"affine_hidden_{index}",
        )(flattened_density)
        value = tf.keras.layers.Activation("softplus", name=f"softplus_{index}")(
            tf.keras.layers.Add(name=f"hidden_sum_{index}")([convex_term, affine_term])
        )

    convex_energy = tf.keras.layers.Dense(
        1,
        use_bias=False,
        kernel_constraint=tf.keras.constraints.NonNeg(),
        kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.05),
        kernel_regularizer=regularizer,
        name="convex_energy",
    )(value)
    affine_energy = tf.keras.layers.Dense(1, kernel_regularizer=regularizer, name="affine_energy")(
        flattened_density
    )
    kinetic_energy = tf.keras.layers.Lambda(
        lambda tensors: tf.squeeze(tensors[0] + tensors[1], axis=-1), name="kinetic_energy"
    )([convex_energy, affine_energy])

    return tf.keras.Model(inputs={"density": density}, outputs={"kinetic_energy": kinetic_energy})
