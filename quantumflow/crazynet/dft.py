import tensorflow as tf
import numpy as np

import quantumflow

from .transformer import CrazyNet, TFWhileCrazyNet
from quantumflow.noninteracting_1d import IntegrateLayer


class XLayer(tf.keras.layers.Layer):
    
    def __init__(self, dataset, subsample_inputs=1, **kwargs):
        super().__init__(**kwargs)
        self.subsample_inputs = subsample_inputs
        
        self.x = tf.Variable(initial_value=np.float32(dataset.x[np.newaxis, :, np.newaxis]), trainable=False, name='x')
        self.x_inputs = tf.Variable(initial_value=np.float32(dataset.x[np.newaxis, np.newaxis, ::subsample_inputs, np.newaxis]), trainable=False, name='x_inputs')

    def call(self, density):
        return (
            tf.repeat(self.x, tf.shape(density)[0], axis=0),
            tf.repeat(tf.repeat(self.x_inputs, self.x.shape[1], axis=1), tf.shape(density)[0], axis=0),
            tf.repeat(tf.expand_dims(tf.expand_dims(density[:, ::self.subsample_inputs], axis=-1), axis=1), self.x.shape[1], axis=1)
        )
    
    def get_config(self):
        return {"subsample_inputs": self.subsample_inputs}

    
    
def CrazyNet_KineticEnergyDensityFunctional(run_dir, dataset, subsample_inputs=1, loop=False, **kwargs):    
    density = tf.keras.layers.Input(shape=dataset.density.shape[1:], name='density')
    
    x, x_inputs, inputs = XLayer(dataset, subsample_inputs)(density)

    if loop:
        value = TFWhileCrazyNet(num_outputs=1, **kwargs)(x, x_inputs, inputs)
    else:
        value = CrazyNet(num_outputs=1, **kwargs)(x, x_inputs, inputs)
    
    kinetic_energy_density = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='kinetic_energy_density')(value)
    kinetic_energy = IntegrateLayer(dataset.h, name='kinetic_energy')(kinetic_energy_density)
    
    return tf.keras.Model(inputs={'density': density}, outputs={'kinetic_energy': kinetic_energy, 'kinetic_energy_density': kinetic_energy_density})
