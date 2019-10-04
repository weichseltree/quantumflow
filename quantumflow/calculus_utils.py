import numpy as np


def integrate(data, h, axis=-1):
    if data.shape[axis] < 2:
        raise ValueError(
            "Integration failed: time-axis {} has {} elements, required: >=2".format(axis, data.shape[axis]))
    return h * (np.sum(data, axis=axis) - 0.5 * (np.take(data, 0, axis=axis) + np.take(data, -1, axis=axis)))


def integrate_simpson(data, h, axis=-1):
    if data.shape[axis] < 2:
        raise ValueError(
            "Integration failed: time-axis {} has {} elements, required: >=2".format(axis, data.shape[axis]))
    integral = 0
    if not (data.shape[axis] > 2 and data.shape[axis] % 2 == 1):
        integral = integrate(np.take(data, [-2, -1], axis=axis), h, axis)
        if data.shape[axis] == 2:
            return integral
        data = np.take(data, range(0, data.shape[axis] - 1), axis=axis)

    even = np.take(data, range(0, data.shape[axis], 2), axis=axis)
    odd = np.take(data, range(1, data.shape[axis], 2), axis=axis)

    return integral + h / 3 * (2 * np.sum(even, axis=axis) + 4 * np.sum(odd, axis=axis) - np.take(data, 0, axis=axis)
                                                                                        - np.take(data, -1, axis=axis))

def laplace(data, h):  # time_axis=1
    temp_laplace = 1 / h ** 2 * (data[:, :-2, :] + data[:, 2:, :] - 2 * data[:, 1:-1, :])
    return np.pad(temp_laplace, ((0, 0), (1, 1), (0, 0)), 'constant')

def normalize(function, h, axis=-1):   
    norm = integrate_simpson(function, h, axis=axis)
    return function * 1 / np.expand_dims(norm, axis=axis)

def rbf_kernel(X, X_train, gamma):
    return np.exp(-gamma*np.sum(np.square(X[:, :, np.newaxis] - np.transpose(X_train)[np.newaxis, :, :]), 1))

def predict(X, X_train, weights, gamma):
    return np.sum(weights[np.newaxis, :]*rbf_kernel(X, X_train, gamma), 1)

def functional_derivative(X, X_train, weights, gamma, h):
    return -1/h*np.sum(weights[np.newaxis, :]*2*gamma*(X[:, :, np.newaxis] - np.transpose(X_train)[np.newaxis, :, :])*rbf_kernel(X, X_train, gamma)[:, np.newaxis, :], 2)
