import tensorflow as tf


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

