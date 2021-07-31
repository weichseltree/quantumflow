import tensorflow as tf
import numpy as np

import quantumflow


def CrazyNet_KineticEnergyDensityFunctional(run_dir, dataset, **kwargs):
    import quantumflow.crazynet
    
    density = tf.keras.layers.Input(shape=dataset.density.shape[1:], name='density')
    
    inputs = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(density) # (batch_size, grid, 1)
    x_inputs = tf.expand_dims(tf.expand_dims(dataset.x, axis=0), axis=-1) # (1, grid, 1)
    
    # TODO: extend CrazyNet to take inputs with 2 batch dims
    
    quantumflow.crazynet.CrazyNet(**kwargs)
    
    kinetic_energy_density = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='kinetic_energy_density')(value)
    kinetic_energy = IntegrateLayer(dataset.h, name='kinetic_energy')(kinetic_energy_density)
    
    return tf.keras.Model(inputs={'density': density}, outputs={'kinetic_energy': kinetic_energy, 'kinetic_energy_density': kinetic_energy_density})

