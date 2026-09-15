"""Input-convex kinetic-energy functionals implemented with JAX."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import optax

Array = jax.Array
Parameters = dict[str, tuple[Array, Array]]


def init_icnn(key: Array, input_size: int, hidden_units: Sequence[int] = (128, 128)) -> Parameters:
    """Initialize an ICNN parameter tree for densities of ``input_size`` points."""
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    if not hidden_units or any(units <= 0 for units in hidden_units):
        raise ValueError("hidden_units must contain one or more positive integers")

    output_sizes = (*hidden_units, 1)
    keys = jax.random.split(key, len(output_sizes) + len(hidden_units))
    params: Parameters = {}
    previous_size = input_size
    for index, outputs in enumerate(output_sizes):
        affine_key = keys[index]
        affine_weight = jax.random.normal(affine_key, (input_size, outputs)) / jnp.sqrt(input_size)
        bias = jnp.zeros((outputs,))
        params[f"affine_{index}"] = (affine_weight, bias)
        if index:
            convex_key = keys[len(output_sizes) + index - 1]
            convex_weight = jax.random.normal(convex_key, (previous_size, outputs)) / jnp.sqrt(
                previous_size
            )
            params[f"convex_{index}"] = (convex_weight, jnp.zeros((outputs,)))
        previous_size = outputs
    return params


def _positive(value: Array) -> Array:
    return jax.nn.softplus(value)


def kinetic_energy(params: Parameters, density: Array) -> Array:
    """Evaluate a convex kinetic-energy functional for one or more densities.

    ``density`` is shaped ``(..., grid_points)``. Hidden-to-hidden and output
    weights are reparameterized by Softplus, ensuring non-negative coefficients.
    Combined with Softplus activations, this guarantees convexity in the input.
    """
    density = jnp.asarray(density)
    value = None
    layer_count = len([key for key in params if key.startswith("affine_")])
    for index in range(layer_count):
        affine_weight, bias = params[f"affine_{index}"]
        value_from_density = density @ affine_weight + bias
        if index:
            convex_weight, convex_bias = params[f"convex_{index}"]
            value = value_from_density + value @ _positive(convex_weight) + convex_bias
        else:
            value = value_from_density
        if index < layer_count - 1:
            value = jax.nn.softplus(value)
        else:
            value = jnp.squeeze(value, axis=-1)
    return value


def functional_derivative(params: Parameters, density: Array) -> Array:
    """Compute ``delta T[n] / delta n`` independently for each batch member."""
    density = jnp.asarray(density)
    if density.ndim == 1:
        return jax.grad(lambda values: kinetic_energy(params, values))(density)
    return jax.vmap(jax.grad(lambda values: kinetic_energy(params, values)))(density)


def potential_from_kinetic_derivative(
    kinetic_derivative: Array, chemical_potential: Array | float
) -> Array:
    """Recover ``v = mu - delta T[n] / delta n`` from the Euler equation."""
    return jnp.asarray(chemical_potential) - jnp.asarray(kinetic_derivative)


def training_step(
    params: Parameters,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    density: Array,
    target_energy: Array,
) -> tuple[Parameters, optax.OptState, Array]:
    """Perform one Optax update minimizing mean-squared kinetic-energy error."""

    def loss_function(current_params: Parameters) -> Array:
        return jnp.mean(jnp.square(kinetic_energy(current_params, density) - target_energy))

    loss, gradients = jax.value_and_grad(loss_function)(params)
    updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
    return optax.apply_updates(params, updates), optimizer_state, loss


def make_training_step(optimizer: optax.GradientTransformation):
    """Create a JIT-compiled training step for a fixed Optax optimizer."""

    @jax.jit
    def step(
        params: Parameters,
        optimizer_state: optax.OptState,
        density: Array,
        target_energy: Array,
    ) -> tuple[Parameters, optax.OptState, Array]:
        return training_step(params, optimizer, optimizer_state, density, target_energy)

    return step
