import tensorflow as tf
import os
import pickle
from quantumflow.colab_train_utils import load_hyperparameters
from quantumflow.numerov_solver import solve_schroedinger

@tf.function
def generate_potentials(return_x=False,
                        return_h=False,
                        dataset_size=100, 
                        discretisation_points=500, 
                        n_gauss=3, 
                        interval_length=1.0,
                        a_minmax=(0.0, 3*10.0), 
                        b_minmax=(0.4, 0.6), 
                        c_minmax=(0.03, 0.1), 
                        n_method='sum',
                        dtype='float64',
                        **kwargs):
    
    if dtype == 'double' or dtype == 'float64':
        dtype = tf.float64
    elif dtype == 'float':
        dtype = tf.float32
    else:
        raise ValueError('unknown dtype {}'.format(dtype))
    

    x = tf.linspace(tf.constant(0.0, dtype=dtype), interval_length, discretisation_points, name="x")

    a = tf.random.uniform((dataset_size, 1, n_gauss), minval=a_minmax[0], maxval=a_minmax[1], dtype=dtype, name="a")
    b = tf.random.uniform((dataset_size, 1, n_gauss), minval=b_minmax[0]*interval_length, maxval=b_minmax[1]*interval_length, dtype=dtype, name="b")
    c = tf.random.uniform((dataset_size, 1, n_gauss), minval=c_minmax[0]*interval_length, maxval=c_minmax[1]*interval_length, dtype=dtype, name="c")

    curves = -tf.square(tf.expand_dims(tf.expand_dims(x, 0), 2) - b)/(2*tf.square(c))
    curves = -a*tf.exp(curves)

    if n_method == 'sum':
        potentials = tf.reduce_sum(curves, -1, name="potentials")
    elif n_method == 'mean':
        potentials = tf.reduce_mean(curves, -1, name="potentials")
    else:
        raise NotImplementedError('Method {} is not implemented.'.format(n_method))

    returns = [potentials]

    if return_x:
        returns += [x]
    
    if return_h:
        h = tf.cast(interval_length/(discretisation_points-1), dtype=dtype) # discretisation interval
        returns += [h]
   
    return returns

def generate_datasets(data_dir, experiment, generate_names):
    if not isinstance(generate_names, list):
        generate_names = [generate_names]

    base_dir = os.path.join(data_dir, experiment)
    file_hyperparams = os.path.join(base_dir, "hyperparams.config")

    for run_name in generate_names:
        params = load_hyperparameters(file_hyperparams, run_name=run_name, globals=globals())

        tf.set_random_seed(params['seed'])
        potential, x, h = generate_potentials(return_x=True, return_h=True, **params)

        params['h'] = h
        energies, wavefunctions = solve_schroedinger(potential, params)
        
        with open(os.path.join(base_dir, params['filename'] + '.pkl'), 'wb') as f:
            pickle.dump({'x': x.numpy(), 'h': h.numpy(), 'potential': potential.numpy(), 'wavefunctions': wavefunctions.numpy(), 'energies': energies.numpy()}, f)

        print("dataset", params['filename'] + '.pkl', "saved to", base_dir)


def integrate(y, h):
    return h*tf.reduce_sum((y[:, :-1] + y[:, 1:])/2., axis=1, name='trapezoidal_integral_approx')

    
def laplace(data, h):  # time_axis=1
    temp_laplace = 1 / h ** 2 * (data[:, :-2, :] + data[:, 2:, :] - 2 * data[:, 1:-1, :])
    return tf.pad(temp_laplace, ((0, 0), (1, 1), (0, 0)), 'constant')