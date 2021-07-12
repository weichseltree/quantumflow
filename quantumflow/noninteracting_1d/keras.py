import tensorflow as tf

import quantumflow

class KineticEnergyFunctionalDerivativeModel(tf.keras.Model):
    def __init__(self, base_model, dataset, run_dir):
        super().__init__()
        self.base_model = quantumflow.instantiate(base_model, dataset=dataset, run_dir=run_dir)
        self.h = tf.constant(dataset.h, dtype=dataset.dtype)

        self.output_names = sorted(['derivative'] + self.base_model.output_names)
        self.input_names = self.base_model.input_names

    @tf.function
    def call(self, density):
        density = tf.nest.flatten(density)

        with tf.GradientTape() as tape:
            tape.watch(density)
            outputs = self.base_model(density)

        outputs['derivative'] = tf.identity(1/self.h*tape.gradient(outputs['kinetic_energy'], density), name='derivative')
        return outputs

    def _set_output_attrs(self, outputs):
        super()._set_output_attrs(outputs)
        self.output_names = sorted(['derivative'] + self.base_model.output_names)

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
