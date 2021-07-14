import tensorflow as tf
import numpy as np

import quantumflow


class IntegrateLayer(tf.keras.layers.Layer):
    def __init__(self, h=1.0, **kwargs):
        super().__init__(**kwargs)
        self.h = h

    def call(self, inputs):
        return self.h*tf.reduce_sum((inputs[:, :-1] + inputs[:, 1:])/2., axis=1, name='trapezoidal_integral_approx')

    def get_config(self):
        config = super().get_config()
        config.update({'h': self.h})
        return config


def ResNet_KineticEnergyDensityFunctional(run_dir, dataset, blocks, l2_regularisation=0.0, **kwargs):
    kernel_regularizer = tf.keras.regularizers.l2(l2_regularisation) if l2_regularisation > 0.0 else None

    density = tf.keras.layers.Input(shape=dataset.density.shape[1:], name='density')
    value = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(density)
    
    def resnet_block(input, filters, kernel_size, padding=None, activation=None, add_input=True):
        value = input
        for layer in range(len(filters)):
            value = tf.keras.layers.Conv1D(filters=filters[layer], 
                                        kernel_size=kernel_size[layer], 
                                        padding=padding,
                                        use_bias=True,
                                        activation=activation if layer < len(filters)-1 else None,
                                        kernel_regularizer=kernel_regularizer)(value)

        if add_input:
            value = tf.keras.layers.Add()([value, input])
        
        return tf.keras.layers.Activation(activation=activation)(value) if activation is not None else value

    for layer in range(len(blocks)):
        value = resnet_block(value, **blocks[layer])

    kinetic_energy_density = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='kinetic_energy_density')(value)
    kinetic_energy = IntegrateLayer(dataset.h, name='kinetic_energy')(kinetic_energy_density)
    
    return tf.keras.Model(inputs={'density': density}, outputs={'kinetic_energy': kinetic_energy, 'kinetic_energy_density': kinetic_energy_density})


class MultiplierLayer(tf.keras.layers.Layer):
    def __init__(self, scale_init=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale_init = scale_init
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            'scale',
            shape=input_shape[-1],
            initializer=tf.keras.initializers.Constant(value=self.scale_init),
            trainable=True)

    def get_config(self):
        config = super().get_config()
        config.update({'scale_init': self.scale_init})
        return config
    
    def call(self, inputs):
        return inputs * self.scale
    
    
class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, bias_init=0.0, **kwargs):
        super().__init__(**kwargs)
        self.bias_init = bias_init

    def build(self, input_shape):
        self.bias = self.add_weight(
            'bias',
            shape=input_shape[-1],
            initializer=tf.keras.initializers.Constant(value=self.bias_init),
            trainable=True)
        
    def get_config(self):
        config = super().get_config()
        config.update({'bias_init': self.bias_init})
        return config
    
    def call(self, inputs):
        return inputs + self.bias


def FixupResNet_KineticEnergyDensityFunctional(run_dir, dataset, blocks, final_layer, l2_regularisation=0.0, fixup_m=3, **kwargs):
    kernel_regularizer = tf.keras.regularizers.l2(l2_regularisation) if l2_regularisation > 0.0 else None

    density = tf.keras.layers.Input(shape=dataset.density.shape[1:], name='density')
    value = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.pad(x, [(0, 0), (450, 450)]), axis=-1))(density)
    
    fixup_L = len(blocks)    
    fixup_scale_variance = fixup_L**(-1 / (fixup_m - 1))
        
    def fixup_bottleneck_block(input, filters, kernel_size, padding=None, activation=None, add_input=True):
        value = input
        input = tf.keras.layers.Lambda(lambda x: x[:, 50:-50, :])(input)
        value = BiasLayer()(value)
        
        for layer in range(len(filters)):
            
            if layer < len(filters)-1:
                kernel_initializer = tf.keras.initializers.VarianceScaling(
                    scale=2 * fixup_scale_variance,                          
                    mode='fan_in', 
                    distribution='truncated_normal')
            else:
                kernel_initializer = tf.keras.initializers.Zeros()
            
            value = tf.keras.layers.Conv1D(
                filters=filters[layer], 
                kernel_size=kernel_size[layer], 
                padding=padding,
                kernel_initializer=kernel_initializer,
                use_bias=True,
                activation=activation,
                kernel_regularizer=kernel_regularizer)(value)
            
        value = MultiplierLayer()(value)
        value = BiasLayer()(value)

        if add_input:
            value = tf.keras.layers.Add()([value, input])
        
        return tf.keras.layers.Activation(activation=activation)(value) if activation is not None else value

    for layer in range(len(blocks)):
        value = fixup_bottleneck_block(value, **blocks[layer])
        
    value = BiasLayer()(value)
    value = tf.keras.layers.Conv1D(
        kernel_initializer=tf.keras.initializers.Zeros(),
        kernel_regularizer=kernel_regularizer, **final_layer)(value)

    kinetic_energy_density = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='kinetic_energy_density')(value)
    kinetic_energy = IntegrateLayer(dataset.h, name='kinetic_energy')(kinetic_energy_density)
    
    return tf.keras.Model(inputs={'density': density}, outputs={'kinetic_energy': kinetic_energy, 'kinetic_energy_density': kinetic_energy_density})

