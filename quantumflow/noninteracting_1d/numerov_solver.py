import tensorflow as tf
from quantumflow.utils import integrate

# recurrent tensorflow cell for solving the numerov equation recursively
class ShootingNumerovCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, shape, h, **kwargs):
        super(ShootingNumerovCell, self).__init__(autocast=False, **kwargs)
        self._h2_scaled = 1 / 12 * h ** 2
        self.shape = shape

    @property
    def state_size(self):
        return self.shape + (4,)

    @property
    def output_size(self):
        return self.shape + (1,)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        k_m2, k_m1, y_m2, y_m1 = tf.unstack(states[0], axis=-1)
        
        y = (2 * (1 - 5 * self._h2_scaled * k_m1) * y_m1 - (1 + self._h2_scaled * k_m2) * y_m2) / (1 + self._h2_scaled * inputs)

        new_state = tf.stack([k_m1, inputs, y_m1, y], axis=-1)
        return y, new_state

# tf function for using the shooting numerov method
#
# the numerov_init_slope is the slope of the solution at x=0
# it can be constant>0 because it's actual value will be determined when the wavefunction is normalized
#
def shooting_numerov(k_squared, h, numerov_init_slope, dtype):
    init_values = tf.zeros_like(k_squared[:, 0])
    one_step_values = numerov_init_slope * h * tf.ones_like(k_squared[:, 0])
    init_state = tf.stack([k_squared[:, 0], k_squared[:, 1], init_values, one_step_values], axis=-1)
    outputs = tf.keras.layers.RNN(ShootingNumerovCell(k_squared.shape[2:], h), return_sequences=True, dtype=dtype)(k_squared[:, 2:], initial_state=init_state)
    output = tf.concat([tf.expand_dims(init_values, axis=1), tf.expand_dims(one_step_values, axis=1), outputs], axis=1)
    return output


# returns the rearranged schroedinger equation term in the numerov equation
# k_squared = 2*m_e/h_bar**2*(E - V(x))
def numerov_k_squared(potentials, energies):
    return 2 * (tf.expand_dims(energies, axis=1) - tf.repeat(tf.expand_dims(potentials, axis=2), energies.shape[1], axis=2))


@tf.function
def find_split_energies(potentials, N, h, dtype, numerov_init_slope):
    M = potentials.shape[0]

    # Knotensatz: number of roots = quantum state
    # so target root = target excited state quantum number
    target_roots = tf.repeat(tf.expand_dims(tf.range(N + 1), axis=0), M, axis=0)

    # lowest value of potential as lower bound
    E_split = tf.repeat(tf.expand_dims(tf.reduce_min(potentials, axis=1), axis=1), N + 1, axis=1)

    solutions_split = tf.zeros((potentials.shape[0], potentials.shape[1], N + 1), dtype=potentials.dtype)
    not_converged = tf.ones(potentials.shape[0], dtype=tf.bool)
    search_boost = tf.ones_like(E_split, dtype=tf.bool)
    E_delta = tf.ones_like(E_split)

    while tf.math.reduce_any(not_converged):
        V_split = numerov_k_squared(tf.boolean_mask(potentials, not_converged), tf.boolean_mask(E_split, not_converged))

        solutions_split_new = shooting_numerov(V_split, h, numerov_init_slope, dtype)

        partitioned_data = tf.dynamic_partition(solutions_split, tf.cast(not_converged, tf.int32) , 2)
        condition_indices = tf.dynamic_partition(tf.range(tf.shape(solutions_split)[0]), tf.cast(not_converged, tf.int32) , 2)

        solutions_split = tf.dynamic_stitch(condition_indices, [partitioned_data[0], solutions_split_new])
        solutions_split.set_shape((potentials.shape[0], potentials.shape[1], N + 1))

        roots_split = tf.reduce_sum(tf.cast(detect_roots(solutions_split), tf.int32), axis=1)

        not_converged = tf.logical_and(tf.logical_not(tf.reduce_all(tf.equal(roots_split, target_roots), axis=1)), not_converged)

        search_direction = tf.cast(roots_split < target_roots, potentials.dtype) - tf.cast(roots_split > target_roots, potentials.dtype)
        boost = tf.logical_and(tf.equal(search_direction, tf.sign(E_delta)), search_boost)

        E_delta += tf.cast(boost, potentials.dtype)*E_delta
        stop_boost = search_direction * tf.sign(E_delta) < 0
        search_boost &= tf.logical_not(stop_boost)
        E_delta += -1.5*E_delta*tf.cast(stop_boost, potentials.dtype)

        E_split += E_delta*tf.expand_dims(tf.cast(not_converged, potentials.dtype), axis=-1)

    return E_split


def detect_roots(array):
    return tf.logical_or(tf.equal(array[:, 1:], 0), array[:, 1:] * array[:, :-1] < 0)

@tf.function
def solve_numerov(potentials, target_roots, split_energies, h, numerov_init_slope, dtype):
    E_low = split_energies[:, :-1]
    E_high = split_energies[:, 1:]

    # because the search interval is halved at every step
    # 32 iterations will always converge to the best numerically possible accuracy of E
    # (empirically ~25 steps)

    E = 0.5 * (E_low + E_high)
    E_last = E * 2
    
    while tf.reduce_any(tf.logical_not(tf.equal(E_last, E))):
        V = numerov_k_squared(potentials, E)

        solutions = shooting_numerov(V, h, numerov_init_slope, dtype)
        roots = tf.reduce_sum(tf.cast(detect_roots(solutions), tf.int32), axis=1)

        update_low = roots <= target_roots
        update_high = tf.logical_not(update_low)

        E_low = tf.where(update_low, E, E_low)
        E_high = tf.where(update_high, E, E_high)

        E_last = E
        E = 0.5 * (E_low + E_high)

    solutions_low = shooting_numerov(numerov_k_squared(potentials, E_low), h, numerov_init_slope, dtype)
    roots_low = tf.cast(detect_roots(solutions_low), tf.double)

    solutions_high = shooting_numerov(numerov_k_squared(potentials, E_high), h, numerov_init_slope, dtype)
    roots_high = tf.cast(detect_roots(solutions_high), tf.double)

    roots_diff = tf.abs(roots_high - roots_low)  

    roots_cumsum = tf.cumsum(tf.pad(roots_diff, ((0, 0), (1, 0), (0, 0)), 'constant'), axis=1)

    invalid = tf.equal(roots_cumsum, tf.expand_dims(roots_cumsum[:, -1], axis=1))

    return solutions_low, E, invalid


@tf.function
def solve_schroedinger(potentials, N, h, dtype, numerov_init_slope):
    M = potentials.shape[0]
    G = potentials.shape[1]
    
    E_split = find_split_energies(potentials, N, h, dtype, numerov_init_slope)

    target_roots = tf.repeat(tf.expand_dims(tf.range(N), axis=0), M, axis=0)
    solutions_forward, E_forward, invalid_forward = solve_numerov(potentials, target_roots, E_split, h, numerov_init_slope, dtype)
    #solutions_forward /= tf.expand_dims(tf.reduce_max(tf.abs(solutions_forward)*tf.cast(tf.logical_not(invalid_forward), tf.double), axis=1), axis=1)

    solutions_backward, E_backward, invalid_backward = solve_numerov(tf.reverse(potentials, axis=[1]), target_roots, E_split, h, numerov_init_slope, dtype)
    solutions_backward = tf.reverse(solutions_backward, axis=[1])
    invalid_backward = tf.reverse(invalid_backward, axis=[1])
    #solutions_backward /= tf.expand_dims(tf.reduce_max(tf.abs(solutions_backward)*tf.cast(tf.logical_not(invalid_backward), tf.double), axis=1), axis=1)

    n_invalid_forward = tf.reduce_sum(tf.cast(invalid_forward, tf.int32), axis=1)
    n_invalid_backward = tf.reduce_sum(tf.cast(invalid_backward, tf.int32), axis=1)
    merge_index = (G - n_invalid_forward - n_invalid_backward)//2 + n_invalid_forward

    merge_value_forward = tf.reduce_sum(tf.gather(tf.transpose(solutions_forward, perm=[0, 2, 1]), tf.expand_dims(merge_index, axis=2), batch_dims=2), axis=2)
    merge_value_backward = tf.reduce_sum(tf.gather(tf.transpose(solutions_backward, perm=[0, 2, 1]), tf.expand_dims(merge_index, axis=2), batch_dims=2), axis=2)

    factor = merge_value_forward/merge_value_backward
    solutions_backward *= tf.expand_dims(factor, axis=1)

    join_mask = tf.expand_dims(tf.expand_dims(tf.range(G), axis=0), axis=2) < tf.expand_dims(merge_index, axis=1)

    solutions = tf.where(join_mask, solutions_forward, solutions_backward)

    #normalization
    density = solutions ** 2
    norm = integrate(density, h)
    solutions *= 1 / tf.sqrt(tf.expand_dims(norm, axis=1))

    E = 0.5*(E_forward + E_backward)
    
    return E, solutions