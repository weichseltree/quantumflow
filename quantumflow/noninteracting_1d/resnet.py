import tensorflow as tf
import numpy as np

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


def CNN_KineticEnergyFunctional(params):
    kernel_regularizer = tf.keras.regularizers.l2(params['model_kwargs']['l2_regularisation']) if params['model_kwargs'].get('l2_regularisation', 0.0) > 0.0 else None
    bias_initializer = tf.constant_initializer(value=params['dataset']['targets_mean']['kinetic_energy']) if params['model_kwargs'].get('bias_mean_initialisation', False) else None
    conv1d_layer = SymmetricConv1D if params['model_kwargs'].get('symmetric', False) else tf.keras.layers.Conv1D

    density = tf.keras.layers.Input(shape=params['dataset']['features_shape']['density'], name='density')
    value = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(density)
    
    for layer in range(len(params['model_kwargs']['filters'])):
        value = conv1d_layer(filters=params['model_kwargs']['filters'][layer], 
                             kernel_size=params['model_kwargs']['kernel_size'][layer], 
                             activation=params['model_kwargs']['activation'] if not params['model_kwargs'].get('batch_normalization', False) else None, 
                             padding=params['model_kwargs']['padding'],
                             kernel_regularizer=kernel_regularizer)(value)
        if params['model_kwargs'].get('batch_normalization', False):
            value = tf.keras.layers.BatchNormalization()(value)
            value = tf.keras.layers.Activation(params['model_kwargs']['activation'])(value)

    value = tf.keras.layers.Flatten()(value)
    value = tf.keras.layers.Dense(1, kernel_regularizer=kernel_regularizer, bias_initializer=bias_initializer)(value)
    kinetic_energy = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='kinetic_energy')(value)

    return tf.keras.Model(inputs={'density': density}, outputs={'kinetic_energy': kinetic_energy})


def CNN_KineticEnergyDensityFunctional(params):
    kernel_regularizer = tf.keras.regularizers.l2(params['model_kwargs']['l2_regularisation']) if params['model_kwargs'].get('l2_regularisation', 0.0) > 0.0 else None
    bias_initializer = tf.constant_initializer(value=params['dataset']['targets_mean']['kinetic_energy']) if params['model_kwargs'].get('bias_mean_initialisation', False) else None

    density = tf.keras.layers.Input(shape=params['dataset']['features_shape']['density'], name='density')
    value = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(density)
    
    for layer in range(len(params['model_kwargs']['filters'])):
        value = tf.keras.layers.Conv1D(filters=params['model_kwargs']['filters'][layer], 
                                       kernel_size=params['model_kwargs']['kernel_size'][layer], 
                                       activation=params['model_kwargs']['activation'] if not params['model_kwargs'].get('batch_normalization', False) and layer < len(params['model_kwargs']['filters'])-1 else None, 
                                       padding=params['model_kwargs']['padding'],
                                       kernel_regularizer=kernel_regularizer)(value)

        if params['model_kwargs'].get('batch_normalization', False):
            value = tf.keras.layers.BatchNormalization()(value)

            if layer < len(params['model_kwargs']['filters'])-1:
                value = tf.keras.layers.Activation(params['model_kwargs']['activation'])(value)

    kinetic_energy_density = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='kinetic_energy_density')(value)
    kinetic_energy = IntegrateLayer(params['dataset']['h'], name='kinetic_energy')(kinetic_energy_density)
    
    return tf.keras.Model(inputs={'density': density}, outputs={'kinetic_energy': kinetic_energy, 'kinetic_energy_density': kinetic_energy_density})


def ResNet_KineticEnergyDensityFunctional(run_dir, dataset, blocks, l2_regularisation=0.0, bias_mean_initialisation=False, **kwargs):
    kernel_regularizer = tf.keras.regularizers.l2(l2_regularisation) if l2_regularisation > 0.0 else None
    bias_initializer = tf.constant_initializer(value=np.mean(dataset.kinetic_energy)) if bias_mean_initialisation else None

    density = tf.keras.layers.Input(shape=dataset.density.shape[1:], name='density')
    value = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(density)
    
    def resnet_block(input, filters, kernel_size, padding=None, activation=None, add_input=True):
        value = input
        for layer in range(len(filters)):
            value = tf.keras.layers.Conv1D(filters=filters[layer], 
                                        kernel_size=kernel_size[layer], 
                                        padding=padding,
                                        use_bias=True,
                                        activation= activation if layer < len(filters)-1 else None,
                                        kernel_regularizer=kernel_regularizer)(value)

        if add_input:
            value = tf.keras.layers.Add()([value, input])
        
        if activation is not None:
            return tf.keras.layers.Activation(activation=activation)(value)
        else:
            return value

    for layer in range(len(blocks)):
        value = resnet_block(value, **blocks[layer])

    kinetic_energy_density = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='kinetic_energy_density')(value)
    kinetic_energy = IntegrateLayer(dataset.h, name='kinetic_energy')(kinetic_energy_density)
    
    return tf.keras.Model(inputs={'density': density}, outputs={'kinetic_energy': kinetic_energy, 'kinetic_energy_density': kinetic_energy_density})


class KineticEnergyFunctionalDerivativeModel(tf.keras.Model):
    def __init__(self, params):
        super().__init__()
        self.model = params['model_kwargs']['base_model'](params)
        self.h = tf.constant(params['dataset']['h'], dtype=params['dtype'])

        self.output_names = sorted(['derivative'] + self.model.output_names)
        self.input_names = self.model.input_names

    @tf.function
    def call(self, density):
        density = tf.nest.flatten(density)

        with tf.GradientTape() as tape:
            tape.watch(density)
            predictions = self.model(density)
            kinetic_energy = predictions['kinetic_energy']

        predictions['derivative'] = tf.identity(1/self.h*tape.gradient(kinetic_energy, density), name='derivative')
        return predictions

    #def fit(self, y=None, validation_data=None, **kwargs):
    #    if isinstance(y, dict):
    #        y = tf.nest.flatten(y)
    #    
    #    if isinstance(validation_data, (tuple, list)) and isinstance(validation_data[1], dict):
    #        validation_data = (validation_data[0], tf.nest.flatten(validation_data[1]))#
    #
    #    super().fit(y=y, validation_data=validation_data, **kwargs)

    def _set_output_attrs(self, outputs):
        super()._set_output_attrs(outputs)
        self.output_names = sorted(['derivative'] + self.model.output_names)

    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)

    def save(self, *args, **kwargs):
        return self.model.save(*args, **kwargs)

    def save_weights(self, *args, **kwargs):
        self.model.optimizer = self.optimizer
        returns = self.model.save_weights(*args, **kwargs)
        self.model.optimizer = None
        return returns

    def load_weights(self, *args, **kwargs):
        self.model.optimizer = self.optimizer
        returns = self.model.load_weights(*args, **kwargs)
        self.optimizer = self.model.optimizer
        self.model.optimizer = None
        return returns


#from tensorflow_addons.optimizers import RectifiedAdam, Lookahead

#def Ranger(*args, **kwargs):
#    return Lookahead(RectifiedAdam(*args, **kwargs))