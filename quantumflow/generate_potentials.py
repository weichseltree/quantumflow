import tensorflow as tf
import numpy as np

def tf_generate_potentials(dataset_size=2000, points=500, n_gauss=3, length=1.0,
                           a_minmax=(0.0, 3*10.0), b_minmax=(0.4, 0.6), c_minmax=(0.03, 0.1), return_x=False):
    x = tf.linspace(0.0, length, points, name="x")

    a = tf.random_uniform((dataset_size, 1, n_gauss), minval=a_minmax[0], maxval=a_minmax[1], name="a")
    b = tf.random_uniform((dataset_size, 1, n_gauss), minval=b_minmax[0]*length, maxval=b_minmax[1]*length, name="b")
    c = tf.random_uniform((dataset_size, 1, n_gauss), minval=c_minmax[0]*length, maxval=c_minmax[1]*length, name="c")

    curves = -tf.square(tf.expand_dims(tf.expand_dims(x, 0), 2) - b)/(2*tf.square(c))
    curves = -a*tf.exp(curves)

    potentials = tf.reduce_sum(curves, -1, name="potentials")

    if return_x:
        return potentials, x
    else:
        return potentials

def generate_potentials(dataset_size=2000, points=500, n_gauss=3, length=1.0,
                        a_minmax=(0.0, 3*10.0), b_minmax=(0.4, 0.6), c_minmax=(0.03, 0.1), return_x=False, seed=0):
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(seed)
        potentials, x = tf_generate_potentials(dataset_size, points, n_gauss, length,
                                               a_minmax, b_minmax, c_minmax, return_x=True)
        sess = tf.Session(graph=g)
        np_potentials, np_x = sess.run([potentials, x])

    if return_x:
        return np_potentials, np_x
    else:
        return np_potentials

if __name__ == '__main__':
    print("generating 1D gaussian mixture potential function:")
    print("M = 2000 ... dataset size")
    print("G = 500  ... discretization points of interval [0, 1]")
    print("negative sum of 3 gauss functions")
    print("random uniform amplitude (0, 10), mean (0.4, 0.6), and std (0.03, 0.1)")
    print("")

    np_potentials, np_x = generate_potentials(return_x=True)

    print("potentials: ", np_potentials.shape, np_potentials.dtype)
    print("x:", np_x.shape, np_x.dtype)