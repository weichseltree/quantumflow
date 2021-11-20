import tensorflow as tf
import numpy as np
import quantumflow

from .transformer import XdiffPerciever


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
            tf.ensure_shape(tf.repeat(tf.expand_dims(density[:, ::self.subsample_inputs], axis=1), self.x.shape[1], axis=1), (None, self.x.shape[1], self.x.shape[1]//self.subsample_inputs))
        )
    
    def get_config(self):
        return {"subsample_inputs": self.subsample_inputs}
                
    
def XdiffPerciever_KineticEnergyDensityFunctional(run_dir, dataset, **kwargs):    
    density = tf.keras.layers.Input(shape=dataset.density.shape[1:], name='density')
    
    x, x_inputs, inputs = XLayer(dataset, subsample_inputs=1)(density)
    x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-2), name='x_expand')(x)

    value = XdiffPerciever(num_outputs=1, **kwargs)(x, x_inputs, inputs)
    
    kinetic_energy_density = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x[..., 0], axis=-1), name='kinetic_energy_density')(value)
    kinetic_energy = quantumflow.layers.TrapezoidalIntegral1D(dataset.h, name='kinetic_energy')(kinetic_energy_density)
    
    return quantumflow.Model(inputs={'density': density}, outputs={'kinetic_energy': kinetic_energy, 'kinetic_energy_density': kinetic_energy_density})
