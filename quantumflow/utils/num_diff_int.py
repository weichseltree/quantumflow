import tensorflow as tf
import numpy as np


def integrate(y, h):
    return h*tf.reduce_sum((y[:, :-1] + y[:, 1:])/2., axis=1, name='trapezoidal_integral_approx')


def np_integrate(integrand, h, axis=1):
    return h * (np.sum(integrand, axis=axis) - 0.5 * (np.take(integrand, 0, axis=axis) + np.take(integrand, -1, axis=axis)))


def derivative_five_point(value, h):
    return tf.concat([1/(12*h)*(- 25*value[:, 0:1] + 48*value[:, 1:2] - 36*value[:, 2:3] + 16*value[:, 3:4] - 3*value[:, 4:5]),
                      1/(12*h)*( - 3*value[:, 0:1] - 10*value[:, 1:2] + 18*value[:, 2:3] - 6*value[:, 3:4] + value[:, 4:5]), 
                      1/(12*h)*(    value[:, 0:-4] - 8*value[:, 1:-3]                                + 8*value[:, 3:-1] - value[:, 4:]),
                      1/(12*h)*(  - value[:, -5:-4] + 6*value[:, -4:-3] - 18*value[:, -3:-2] + 10*value[:, -2:-1] + 3*value[:, -1:]),
                      1/(12*h)*(3*value[:, -5:-4] - 16*value[:, -4:-3] + 36*value[:, -3:-2] - 48*value[:, -2:-1] + 25*value[:, -1:])], axis=1)


def laplace_five_point(value, h):
    return tf.concat([1/(h**2)*(2*value[:, 0:1] - 5*value[:, 1:2] + 4*value[:, 2:3] - 1*value[:, 3:4]),
                       1/(12*h**2)*(11*value[:, 0:1] - 20*value[:, 1:2] + 6*value[:, 2:3] + 4*value[:, 3:4] - 1*value[:, 4:5]),
                       1/(12*h**2)*(-value[:, 4:] + 16*value[:, 3:-1] - 30*value[:, 2:-2] + 16*value[:, 1:-3] - value[:, 0:-4]),
                       1/(12*h**2)*(11*value[:, -1:] - 20*value[:, -2:-1] + 6*value[:, -3:-2] + 4*value[:, -4:-3] - 1*value[:, -5:-4]),
                       1/(h**2)*(2*value[:, -1:] - 5*value[:, -2:-1] + 4*value[:, -3:-2] - 1*value[:, -4:-3])], axis=1)


def normalize(function, h):
    integral = integrate(function**2, h)
    function /= tf.sqrt(tf.expand_dims(integral, axis=1))
    return function


def normalize_density(density, h):
    integral = integrate(density, h)
    density /= tf.expand_dims(integral, axis=1)
    return density
