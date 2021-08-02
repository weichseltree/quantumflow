import tensorflow as tf
import numpy as np

import quantumflow

from .transformer import CrazyNet
from quantumflow.noninteracting_1d import IntegrateLayer


def CrazyNet_KineticEnergyDensityFunctional(run_dir, dataset, **kwargs):
    
    density = tf.keras.layers.Input(shape=dataset.density[:, ::10].shape[1:], name='density')
    
    x, x_inputs, inputs = tf.keras.layers.Lambda(
        lambda density: (
            tf.repeat(np.float32(dataset.x[np.newaxis, :, np.newaxis]), tf.shape(density)[0], axis=0),
            tf.repeat(tf.repeat(np.float32(dataset.x[np.newaxis, np.newaxis, ::10, np.newaxis]), dataset.discretisation_points, axis=1), tf.shape(density)[0], axis=0),
            tf.repeat(tf.expand_dims(tf.expand_dims(density, axis=-1), axis=1), dataset.discretisation_points, axis=1)
        )
    )(density)
    
    value = CrazyNet(num_outputs=1, **kwargs)(x, x_inputs, inputs)
    
    kinetic_energy_density = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='kinetic_energy_density')(value)
    kinetic_energy = IntegrateLayer(dataset.h, name='kinetic_energy')(kinetic_energy_density)
    
    return tf.keras.Model(inputs={'density': density}, outputs={'kinetic_energy': kinetic_energy, 'kinetic_energy_density': kinetic_energy_density})

