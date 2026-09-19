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


def save_parameters(params: Parameters, path) -> None:
    """Save an ICNN parameter tree as a portable, non-executable archive.

    Each layer is stored as two arrays under its own name, so the file carries
    no pickled objects and can be read by anything that reads ``.npz``.
    """
    import numpy as np

    arrays = {}
    for name, (weight, bias) in params.items():
        arrays[f"{name}__weight"] = np.asarray(weight)
        arrays[f"{name}__bias"] = np.asarray(bias)
    np.savez_compressed(path, **arrays)


def load_parameters(path) -> Parameters:
    """Load a parameter tree written by :func:`save_parameters`."""
    import numpy as np

    params: Parameters = {}
    with np.load(path, allow_pickle=False) as archive:
        names = {key.rsplit("__", 1)[0] for key in archive.files}
        for name in names:
            params[name] = (
                jnp.asarray(archive[f"{name}__weight"]),
                jnp.asarray(archive[f"{name}__bias"]),
            )
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


def functional_derivative(params: Parameters, density: Array, volume_element: float = 1.0) -> Array:
    """Compute ``delta T[n] / delta n`` independently for each batch member.

    If ``volume_element`` is provided (e.g. ``h**d`` on a grid), the gradient
    is scaled by ``1 / volume_element`` to match the continuum functional derivative.
    """
    density = jnp.asarray(density)
    scale = 1.0 / volume_element
    if density.ndim == 1:
        return scale * jax.grad(lambda values: kinetic_energy(params, values))(density)
    return scale * jax.vmap(jax.grad(lambda values: kinetic_energy(params, values)))(density)


def potential_from_kinetic_derivative(
    kinetic_derivative: Array, chemical_potential: Array | float
) -> Array:
    """Recover ``v = mu - delta T[n] / delta n`` from the Euler equation."""
    return jnp.asarray(chemical_potential) - jnp.asarray(kinetic_derivative)


def reconstruct_potential(
    params: Parameters,
    density: Array,
    chemical_potential: Array | float,
    volume_element: float = 1.0,
) -> Array:
    """Reconstruct the external potential ``v(r)`` from density and chemical potential."""
    derivative = functional_derivative(params, density, volume_element=volume_element)
    return potential_from_kinetic_derivative(derivative, chemical_potential)


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


def composite_training_step(
    params: Parameters,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    density: Array,
    target_energy: Array,
    target_derivative: Array,
    alpha_derivative: float = 1.0,
    volume_element: float = 1.0,
) -> tuple[Parameters, optax.OptState, dict[str, Array]]:
    """Perform one update minimizing joint kinetic energy and functional derivative loss."""

    def loss_function(current_params: Parameters) -> tuple[Array, dict[str, Array]]:
        pred_energy = kinetic_energy(current_params, density)
        loss_energy = jnp.mean(jnp.square(pred_energy - target_energy))

        pred_derivative = functional_derivative(
            current_params, density, volume_element=volume_element
        )
        loss_derivative = jnp.mean(jnp.square(pred_derivative - target_derivative))

        total_loss = loss_energy + alpha_derivative * loss_derivative
        aux = {
            "loss": total_loss,
            "loss_energy": loss_energy,
            "loss_derivative": loss_derivative,
        }
        return total_loss, aux

    (total_loss, aux), gradients = jax.value_and_grad(loss_function, has_aux=True)(params)
    updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
    return optax.apply_updates(params, updates), optimizer_state, aux


def make_composite_training_step(
    optimizer: optax.GradientTransformation,
    alpha_derivative: float = 1.0,
    volume_element: float = 1.0,
):
    """Create a JIT-compiled update step for joint energy + derivative training."""

    @jax.jit
    def step(
        params: Parameters,
        optimizer_state: optax.OptState,
        density: Array,
        target_energy: Array,
        target_derivative: Array,
    ) -> tuple[Parameters, optax.OptState, dict[str, Array]]:
        return composite_training_step(
            params,
            optimizer,
            optimizer_state,
            density,
            target_energy,
            target_derivative,
            alpha_derivative=alpha_derivative,
            volume_element=volume_element,
        )

    return step


def solve_ground_state_density(
    params: Parameters,
    potential: Array,
    num_particles: float,
    volume_element: float = 1.0,
    num_steps: int = 300,
    lr: float = 5e-2,
) -> dict[str, Array | float]:
    """Variational minimization of E[n] = T_s[n] + int v(r) n(r) dr in arbitrary dimensions."""
    v_arr = jnp.asarray(potential, dtype=jnp.float32).reshape(-1)
    grid_size = v_arr.shape[0]

    # Initial parameterization from flat density
    init_logits = jnp.zeros((grid_size,), dtype=jnp.float32)
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(init_logits)

    def logits_to_density(logits: Array) -> Array:
        sp = jax.nn.softplus(logits)
        normalized = sp / (jnp.sum(sp) * volume_element)
        return normalized * num_particles

    def variational_energy(logits: Array) -> Array:
        n = logits_to_density(logits)
        t_s = kinetic_energy(params, n)
        v_ext = jnp.sum(v_arr * n) * volume_element
        return t_s + v_ext

    @jax.jit
    def opt_step(logits: Array, state: optax.OptState) -> tuple[Array, optax.OptState]:
        grads = jax.grad(variational_energy)(logits)
        updates, new_state = optimizer.update(grads, state, logits)
        return optax.apply_updates(logits, updates), new_state

    current_logits = init_logits
    for _ in range(num_steps):
        current_logits, opt_state = opt_step(current_logits, opt_state)

    optimal_density = logits_to_density(current_logits)
    kin_energy = kinetic_energy(params, optimal_density)
    tot_energy = variational_energy(current_logits)
    pot_energy = tot_energy - kin_energy

    # Chemical potential from Euler equation: mu = delta T / delta n + v(r)
    deriv = functional_derivative(params, optimal_density, volume_element=volume_element)
    chemical_potential = jnp.mean(deriv + v_arr)

    return {
        "density": optimal_density,
        "chemical_potential": float(chemical_potential),
        "kinetic_energy": float(kin_energy),
        "potential_energy": float(pot_energy),
        "total_energy": float(tot_energy),
    }


def von_weizsaecker_kinetic_energy(
    density: Array,
    spacing: float = 1.0,
    eps: float = 1e-12,
) -> Array:
    """Evaluate the von Weizsaecker kinetic energy functional:

    T_W[n] = 1/2 int |grad sqrt(n)|^2 dx

    For 1D discretized densities shaped ``(..., grid_points)``.
    Uses central differences on the interior and one-sided differences at boundaries.
    """
    density = jnp.asarray(density)
    sqrt_n = jnp.sqrt(jnp.maximum(density, 0.0) + eps)

    if density.ndim == 1:
        d_sqrt_n_interior = (sqrt_n[2:] - sqrt_n[:-2]) / (2.0 * spacing)
        d_sqrt_n_left = (sqrt_n[1] - sqrt_n[0]) / spacing
        d_sqrt_n_right = (sqrt_n[-1] - sqrt_n[-2]) / spacing
        grad_sqrt_n = jnp.concatenate(
            [d_sqrt_n_left[None], d_sqrt_n_interior, d_sqrt_n_right[None]]
        )
        return 0.5 * jnp.sum(jnp.square(grad_sqrt_n)) * spacing
    else:
        d_sqrt_n_interior = (sqrt_n[:, 2:] - sqrt_n[:, :-2]) / (2.0 * spacing)
        d_sqrt_n_left = (sqrt_n[:, 1:2] - sqrt_n[:, 0:1]) / spacing
        d_sqrt_n_right = (sqrt_n[:, -1:] - sqrt_n[:, -2:-1]) / spacing
        grad_sqrt_n = jnp.concatenate([d_sqrt_n_left, d_sqrt_n_interior, d_sqrt_n_right], axis=-1)
        return 0.5 * jnp.sum(jnp.square(grad_sqrt_n), axis=-1) * spacing


def quantum_potential(
    density: Array,
    spacing: float = 1.0,
    eps: float = 1e-12,
) -> Array:
    """Evaluate the Bohmian quantum potential Q[n] = delta T_W[n] / delta n.

    Q[n](x) = -1/2 (grad^2 sqrt(n)) / sqrt(n)

    This dispersive potential acts as an internal pressure barrier that
    counteracts steep gradient steepening and prevents Navier-Stokes / Euler
    shock wave formation and finite-time blow-ups.
    """
    density = jnp.asarray(density)
    scale = 1.0 / spacing
    if density.ndim == 1:
        return scale * jax.grad(
            lambda n: von_weizsaecker_kinetic_energy(n, spacing=spacing, eps=eps)
        )(density)
    return scale * jax.vmap(
        jax.grad(lambda n: von_weizsaecker_kinetic_energy(n, spacing=spacing, eps=eps))
    )(density)


def regularized_kinetic_energy(
    params: Parameters,
    density: Array,
    spacing: float = 1.0,
    c_w: float = 1.0,
) -> Array:
    """Evaluate regularized kinetic functional: T_s[n] = c_W * T_W[n] + T_corr[n].

    Guarantees the rigorous von Weizsaecker lower bound T_s[n] >= T_W[n] (when c_W=1.0)
    and enforces H^1 Sobolev regularity on the density field.
    """
    t_w = von_weizsaecker_kinetic_energy(density, spacing=spacing)
    t_corr = kinetic_energy(params, density)
    return c_w * t_w + t_corr


def spin_resolved_kinetic_energy(
    params: Parameters,
    density_up: Array,
    density_down: Array,
    spacing: float = 1.0,
    c_w: float = 1.0,
) -> Array:
    """Evaluate spin-resolved kinetic functional via the Oliver-Perdew relation:

    T_s[n_up, n_down] = 1/2 T_s[2 n_up] + 1/2 T_s[2 n_down]
    """
    t_up = regularized_kinetic_energy(params, 2.0 * density_up, spacing=spacing, c_w=c_w)
    t_down = regularized_kinetic_energy(params, 2.0 * density_down, spacing=spacing, c_w=c_w)
    return 0.5 * (t_up + t_down)


def multispecies_kinetic_energy(
    params_or_dict: Parameters | dict[str, Parameters],
    densities: Sequence[Array],
    masses: Sequence[float],
    spacing: float = 1.0,
    c_w: float = 1.0,
) -> Array:
    """Evaluate multi-species mass-weighted kinetic functional:

    T_s[{n_alpha}] = sum_alpha (1 / (2 * m_alpha)) * T_s[n_alpha]
    """
    total = 0.0
    for idx, (n_alpha, m_alpha) in enumerate(zip(densities, masses, strict=True)):
        if isinstance(params_or_dict, dict) and f"species_{idx}" in params_or_dict:
            p = params_or_dict[f"species_{idx}"]
        else:
            p = params_or_dict
        t_alpha = regularized_kinetic_energy(p, n_alpha, spacing=spacing, c_w=c_w)
        total = total + (1.0 / (2.0 * m_alpha)) * t_alpha
    return total


def solve_ground_state_density_stabilized(
    params: Parameters,
    potential: Array,
    num_particles: float,
    volume_element: float = 1.0,
    c_w: float = 1.0,
    num_steps: int = 300,
    lr: float = 5e-2,
) -> dict[str, Array | float]:
    """Variational ground-state search with quantum pressure (T_W) stabilization.

    Minimizes E[n] = c_W * T_W[n] + T_corr[n] + int v(r) n(r) dr with Softplus
    normalization. The dispersive quantum potential prevents charge-sloshing
    and finite-time singular spikes.
    """
    v_arr = jnp.asarray(potential, dtype=jnp.float32).reshape(-1)
    grid_size = v_arr.shape[0]

    init_logits = jnp.zeros((grid_size,), dtype=jnp.float32)
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(init_logits)

    def logits_to_density(logits: Array) -> Array:
        sp = jax.nn.softplus(logits)
        normalized = sp / (jnp.sum(sp) * volume_element)
        return normalized * num_particles

    def variational_energy(logits: Array) -> Array:
        n = logits_to_density(logits)
        t_s = regularized_kinetic_energy(params, n, spacing=volume_element, c_w=c_w)
        v_ext = jnp.sum(v_arr * n) * volume_element
        return t_s + v_ext

    @jax.jit
    def opt_step(logits: Array, state: optax.OptState) -> tuple[Array, optax.OptState]:
        grads = jax.grad(variational_energy)(logits)
        updates, new_state = optimizer.update(grads, state, logits)
        return optax.apply_updates(logits, updates), new_state

    current_logits = init_logits
    for _ in range(num_steps):
        current_logits, opt_state = opt_step(current_logits, opt_state)

    optimal_density = logits_to_density(current_logits)
    kin_energy = regularized_kinetic_energy(
        params, optimal_density, spacing=volume_element, c_w=c_w
    )
    tot_energy = variational_energy(current_logits)
    pot_energy = tot_energy - kin_energy

    # Recover chemical potential from regularized Euler equation: mu = delta T / delta n + v(r)
    scale = 1.0 / volume_element
    deriv = scale * jax.grad(
        lambda n: regularized_kinetic_energy(params, n, spacing=volume_element, c_w=c_w)
    )(optimal_density)
    chemical_potential = jnp.mean(deriv + v_arr)

    return {
        "density": optimal_density,
        "chemical_potential": float(chemical_potential),
        "kinetic_energy": float(kin_energy),
        "potential_energy": float(pot_energy),
        "total_energy": float(tot_energy),
    }
