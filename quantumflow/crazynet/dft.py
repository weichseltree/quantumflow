import tensorflow as tf
import numpy as np

import quantumflow

from .transformer import CrazyNet


def CrazyNet_KineticEnergyDensityFunctional(run_dir, dataset, blocks, final_layer, l2_regularisation=0.0, fixup_m=3, **kwargs):
    kernel_regularizer = tf.keras.regularizers.l2(l2_regularisation) if l2_regularisation > 0.0 else None

    density = tf.keras.layers.Input(shape=dataset.density.shape[1:], name='density')
    value = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(density)
        
    value = BiasLayer()(value)
    value = tf.keras.layers.Conv1D(
        kernel_initializer=tf.keras.initializers.Zeros(),
        kernel_regularizer=kernel_regularizer, **final_layer)(value)

    kinetic_energy_density = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='kinetic_energy_density')(value)
    kinetic_energy = IntegrateLayer(dataset.h, name='kinetic_energy')(kinetic_energy_density)
    
    return tf.keras.Model(inputs={'density': density}, outputs={'kinetic_energy': kinetic_energy, 'kinetic_energy_density': kinetic_energy_density})

