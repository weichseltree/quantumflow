import tensorflow as tf
import quantumflow


def ResNet_KineticEnergyDensityFunctional(run_dir, dataset, blocks, l2_regularisation=0.0, **kwargs):
    kernel_regularizer = tf.keras.regularizers.l2(l2_regularisation) if l2_regularisation > 0.0 else None

    density = tf.keras.layers.Input(shape=dataset.density.shape[1:], name='density')
    value = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(density)
    
    def resnet_block(input, filters, kernel_size, padding=None, activation=None, add_input=True):
        value = input
        for layer in range(len(filters)):
            value = tf.keras.layers.Conv1D(
                filters=filters[layer],
                kernel_size=kernel_size[layer],
                padding=padding,
                use_bias=True,
                activation=activation if layer < len(filters) - 1 else None,
                kernel_regularizer=kernel_regularizer)(value)

        if add_input:
            value = tf.keras.layers.Add()([value, input])
        
        return tf.keras.layers.Activation(activation=activation)(value) if activation is not None else value

    for layer in range(len(blocks)):
        value = resnet_block(value, **blocks[layer])

    kinetic_energy_density = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='kinetic_energy_density')(value)
    kinetic_energy = quantumflow.layers.TrapezoidalIntegral1D(dataset.h, name='kinetic_energy')(kinetic_energy_density)
    
    return tf.keras.Model(inputs={'density': density}, outputs={'kinetic_energy': kinetic_energy, 'kinetic_energy_density': kinetic_energy_density})


def FixupResNet_KineticEnergyDensityFunctional(run_dir, dataset, blocks, final_layer, l2_regularisation=0.0, fixup_m=3, **kwargs):
    kernel_regularizer = tf.keras.regularizers.l2(l2_regularisation) if l2_regularisation > 0.0 else None

    density = tf.keras.layers.Input(shape=dataset.density.shape[1:], name='density')
    value = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(density)
    
    fixup_L = len(blocks)
    fixup_scale_variance = fixup_L**(-1 / (fixup_m - 1))
        
    def fixup_bottleneck_block(input, filters, kernel_size, padding=None, activation=None, add_input=True):
        value = input
        value = quantumflow.layers.BiasLayer()(value)
        
        for layer in range(len(filters)):
            
            if layer < len(filters) - 1:
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
            
        value = quantumflow.layers.MultiplierLayer()(value)
        value = quantumflow.layers.BiasLayer()(value)

        if add_input:
            value = tf.keras.layers.Add()([value, input])
        
        return tf.keras.layers.Activation(activation=activation)(value) if activation is not None else value

    for layer in range(len(blocks)):
        value = fixup_bottleneck_block(value, **blocks[layer])
        
    value = quantumflow.layers.BiasLayer()(value)
    value = tf.keras.layers.Conv1D(
        kernel_initializer=tf.keras.initializers.Zeros(),
        kernel_regularizer=kernel_regularizer, **final_layer)(value)

    kinetic_energy_density = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='kinetic_energy_density')(value)
    kinetic_energy = quantumflow.layers.TrapezoidalIntegral1D(dataset.h, name='kinetic_energy')(kinetic_energy_density)
    
    return tf.keras.Model(inputs={'density': density}, outputs={'kinetic_energy': kinetic_energy, 'kinetic_energy_density': kinetic_energy_density})

