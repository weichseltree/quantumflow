import tensorflow as tf
import numpy as np

from quantumflow.calculus_utils import integrate, integrate_simpson, laplace


def unpack_dataset(N, dataset):
    np_x, np_potentials, np_solutions, np_E = dataset
    np_density = np.sum(np.square(np_solutions)[:, :, :N], axis=-1)
    
    dataset_size, discretization_points, _ = np_solutions.shape
    h = (max(np_x) - min(np_x))/(discretization_points-1)
    
    np_potential = np.expand_dims(np_potentials, axis=2)*np_solutions**2
    np_P = integrate_simpson(np_potential, h, axis=1)
    np_K = np_E - np_P

    kinetic_energy = np.sum(np_K[:, :N], axis=-1)
    
    return np_x, np_potentials, np_solutions, np_E, np_density, kinetic_energy, dataset_size, discretization_points, h    


# recurrent tensorflow cell for solving the numerov equation recursively
class ShootingNumerovCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, h=1.0):
        super().__init__()
        self._h2_scaled = 1 / 12 * h ** 2

    def __call__(self, inputs, state, scope=None):
        k_m2, k_m1, y_m2, y_m1 = tf.unstack(state, axis=-1)

        y = (2 * (1 - 5 * self._h2_scaled * k_m1) * y_m1 - (1 + self._h2_scaled * k_m2) * y_m2) / (
                    1 + self._h2_scaled * inputs)

        new_state = tf.stack([k_m1, inputs, y_m1, y], axis=-1)
        return y, new_state

    @property
    def state_size(self):
        return 4

    @property
    def output_size(self):
        return 1

# tf function for using the shooting numerov method
#
# the init_factor is the slope of the solution at x=0
# it can be constant>0 because it's actual value will be determined when the wavefunction is normalized
#
def shooting_numerov(k_squared, h=1, init_factor=1e-128):
    shooting_cell = ShootingNumerovCell(h=h)
    init_state = tf.stack([k_squared[:, 0], k_squared[:, 1], tf.zeros_like(k_squared[:, 2]),
                           init_factor * h * tf.ones_like(k_squared[:, 3])], axis=-1)
    outputs, _ = tf.nn.static_rnn(shooting_cell, tf.unstack(k_squared, axis=1)[2:], initial_state=init_state)
    output = tf.stack([init_state[:, 2], init_state[:, 3]] + outputs, axis=-1)
    return output

# returns the rearranged schroedinger equation term in the numerov equation
# k_squared = 2*m_e/h_bar**2*(E - V(x))
def numerov_k_squared(potentials, energies):
    return 2 * (np.expand_dims(energies, axis=1) - np.repeat(np.expand_dims(potentials, axis=2), energies.shape[1], axis=2))


def detect_roots(array1):
    return np.logical_or(array1[:, 1:] == 0, array1[:, 1:] * array1[:, :-1] < 0)


class NumerovSolver():
    def __init__(self, G, h):
        self.K_SQUARED = tf.placeholder(tf.float64, shape=(None, G))
        self.solution = shooting_numerov(self.K_SQUARED, h=h)
        self.sess = tf.Session()
        self.h = h
        self.G = G
        
    # functtion to solve the shooting numerov equation for a given tensor of k_squared functions
    # the tensor has to have one dimension for the time along wich to solve the equation
    # all other dimensions will be flattened internally but the return value will be reshaped back
    def run_numerov(self, k_squared, time_axis=-1):
        shape = k_squared.shape[:time_axis] + k_squared.shape[time_axis + 1:]
        flattened = np.reshape(np.moveaxis(k_squared, time_axis, -1), (-1, k_squared.shape[time_axis]))
        flattened_solutions = self.sess.run(self.solution, feed_dict={self.K_SQUARED: flattened})
        solutions = np.reshape(flattened_solutions, shape + (k_squared.shape[time_axis],))
        return np.moveaxis(solutions, -1, time_axis)

    
    def solve_numerov(self, np_potentials, target_roots, split_energies, progress=None):

        np_E_low = split_energies[:, :-1].copy()
        np_E_high = split_energies[:, 1:].copy()

        # because the search interval is halved at every step
        # 32 iterations will always converge to the best numerically possible accuracy of E
        # (empirically ~25 steps)

        np_E = 0.5 * (np_E_low + np_E_high)
        np_E_last = np.copy(np_E) * 2

        
        if progress is not None:
            progress.value = 0
            progress.max = np.prod(np_E.shape)
            progress.description = 'Numerov Pass: '
        
        step = 0
        while np.any(np_E_last - np_E):
            np_V = numerov_k_squared(np_potentials, np_E)
            np_solutions = self.run_numerov(np_V, time_axis=1)
            np_roots = np.sum(detect_roots(np_solutions), axis=1)

            np_E_low[np_roots <= target_roots] = np_E[np_roots <= target_roots]
            np_E_high[np_roots > target_roots] = np_E[np_roots > target_roots]

            np_E_last = np_E
            np_E = 0.5 * (np_E_low + np_E_high)

            if progress is not None:
                progress.value = progress.max - np.sum(np_E_last - np_E != 0)
                progress.description = 'Numerov Pass: ' + str(progress.value) + '/' + str(progress.max)
            step += 1

        np_solutions_low = self.run_numerov(numerov_k_squared(np_potentials, np_E_low), time_axis=1)
        np_roots_low = 1 * detect_roots(np_solutions_low)

        np_solutions_high = self.run_numerov(numerov_k_squared(np_potentials, np_E_high), time_axis=1)
        np_roots_high = 1 * detect_roots(np_solutions_high)

        np_roots_diff = np.abs(np_roots_high - np_roots_low)  # useless but keep it
        # assert(np.all(np.sum(np_roots_diff, axis=1) == 1)) # sometimes roots are at different places!

        np_nan_cumsum = np.cumsum(np.pad(np_roots_diff, ((0, 0), (1, 0), (0, 0)), 'constant'), axis=1)
        np_nan_index = np_nan_cumsum == np.expand_dims(np_nan_cumsum[:, -1], axis=1)

        np_solutions_low[np_nan_index] = np.nan

        return np_solutions_low, np_E, step

    
    def find_split_energies(self, np_potentials, N, progress=None):
        M = np_potentials.shape[0]
        
        # Knotensatz: number of roots = quantum state
        # so target root = target excited state quantum number
        target_roots = np.repeat(np.expand_dims(np.arange(N + 1), axis=0), M, axis=0)

        # lowest value of potential as lower bound
        np_E_split = np.repeat(np.expand_dims(np.min(np_potentials, axis=1), axis=1), N + 1, axis=1)

        np_solutions_split = np.zeros((np_potentials.shape[0], np_potentials.shape[1], N + 1), dtype=np.float64)
        not_converged = np.ones(np_potentials.shape[0], dtype=np.bool)
        search_boost = np.ones_like(np_E_split)
        np_E_delta = np.ones_like(np_E_split)

        if progress is not None:
            progress.value = 0
            progress.max = M
            progress.description = 'Searching Roots:'

        step = 0
        while np.any(not_converged):
            np_V_split = numerov_k_squared(np_potentials[not_converged], np_E_split[not_converged])
            np_solutions_split[not_converged] = self.run_numerov(np_V_split, time_axis=1)
            np_roots_split = np.sum(detect_roots(np_solutions_split), axis=1)

            not_converged[np.all(np_roots_split == target_roots, axis=1)] = False

            search_direction = 1 * (np_roots_split < target_roots) - 1 * (np_roots_split > target_roots)
            np_E_delta[np.logical_and(search_direction == np.sign(np_E_delta), search_boost)] *= 2
            search_boost[search_direction * np.sign(np_E_delta) < 0] = 0
            np_E_delta[search_direction * np.sign(np_E_delta) < 0] *= -0.5

            np_E_split[not_converged] += np_E_delta[not_converged]

            if progress is not None:
                progress.value = progress.max - np.sum(not_converged)
                progress.description = 'Searching Roots: ' + str(progress.value) + '/' + str(progress.max)
            step += 1

        return np_E_split, step

    
    def solve_schroedinger(self, np_potentials, N, progress=None):
        M = np_potentials.shape[0]
        G = np_potentials.shape[1]
        
        assert (G == self.G)
        np_E_split, _ = self.find_split_energies(np_potentials, N, progress=progress)

        target_roots = np.repeat(np.expand_dims(np.arange(N), axis=0), M, axis=0)
        np_solutions_forward, np_E_forward, _ = self.solve_numerov(np_potentials, target_roots, np_E_split, progress=progress)
        np_solutions_forward /= np.expand_dims(np.nanmax(np.abs(np_solutions_forward), axis=1), axis=1)

        assert not np.any(np.all(np.isnan(np_solutions_forward), axis=1))

        np_solutions_backward, np_E_backward, _ = self.solve_numerov(np.flip(np_potentials, axis=1), target_roots, np_E_split, progress=progress)
        np_solutions_backward = np.flip(np_solutions_backward, axis=1)
        np_solutions_backward /= np.expand_dims(np.nanmax(np.abs(np_solutions_backward), axis=1), axis=1)

        assert not np.any(np.all(np.isnan(np_solutions_backward), axis=1))

        np_factor = np_solutions_forward / np_solutions_backward

        assert not np.any(np.all(np.isnan(np_factor), axis=1))

        np_solutions_backward *= np.expand_dims(np.nanmedian(np_factor, axis=1), axis=1)

        join_error = np.nanmin(np.abs(np_solutions_backward - np_solutions_forward), axis=1)

        join_error = np.nanmax(np_solutions_backward / np_solutions_forward, axis=1)

        join_index = np.nanargmin(np.abs(np_solutions_backward - np_solutions_forward), axis=1)

        join_mask = np.expand_dims(np.expand_dims(np.arange(np_solutions_backward.shape[1]), axis=0), axis=2) >= np.expand_dims(join_index, axis=1)

        np_solutions = np_solutions_forward
        np_solutions[join_mask] = np_solutions_backward[join_mask]

        # normalization
        np_norm = np_solutions ** 2
        np_norm = integrate_simpson(np_norm, self.h, axis=1)
        np_solutions *= 1 / np.sqrt(np.expand_dims(np_norm, axis=1))

        assert not np.any(np.all(np.isnan(np_solutions), axis=1))
        
        np_E = 0.5*(np_E_forward + np_E_backward)
        
        return np_E, np_solutions
    