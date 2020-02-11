import tensorflow as tf

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
            kinetic_energy = self.model(density)

        derivative = tf.identity(1/self.h*tape.gradient(kinetic_energy, density), name='derivative')
        return derivative, kinetic_energy

    def fit(self, y=None, validation_data=None, **kwargs):
        if isinstance(y, dict):
            y = tf.nest.flatten(y)
        
        if isinstance(validation_data, (tuple, list)) and isinstance(validation_data[1], dict):
            validation_data = (validation_data[0], tf.nest.flatten(validation_data[1]))

        super().fit(y=y, validation_data=validation_data, **kwargs)

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

import time

class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, *args, metrics_freq=0, learning_rate=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_time = None
        self.last_step = None
        self.learning_rate = learning_rate
        self.metrics_freq = metrics_freq

    def on_epoch_end(self, epoch, logs=None):
        """Runs metrics and histogram summaries at epoch end."""
        if self.metrics_freq and epoch % self.metrics_freq == 0 or any(['val_' in key for key in logs.keys()]):
            self._log_metrics(logs, prefix='epoch_', step=epoch)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_weights(epoch)

        if self.embeddings_freq and epoch % self.embeddings_freq == 0:
            self._log_embeddings(epoch)

    def _log_metrics(self, logs, prefix, step):
        
        if self.last_time is not None:
            new_time = time.time()
            logs['epochs_per_second'] = (step - self.last_step)/(new_time - self.last_time)
            self.last_time = new_time
            self.last_step = step
        else:
            self.last_time = time.time()
            self.last_step = step

        if self.learning_rate is not None and isinstance(self.learning_rate, float):
            logs['learning_rate'] = self.learning_rate
        elif isinstance(self.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            logs['learning_rate'] = self.learning_rate(self.model.optimizer.iterations)

        def rename_key(key):
            prepend_val = False
            if 'val_' in key:
                prepend_val = True
                key = key.replace('val_', '')
            if '_loss' in key:
                key = 'loss/' + key.replace('_loss', '')
            if '_mean_absolute_error' in key:
                key = 'mean_absolute_error/' + key.replace('_mean_absolute_error', '')
            if prepend_val:
                key = 'val_' + key
            return key

        logs = {rename_key(key): value for key, value in logs.items()}
        super()._log_metrics(logs, '', step)


class WarmupExponentialDecay(tf.keras.optimizers.schedules.ExponentialDecay):
    def __init__(self, warmup_steps=None, cold_steps=None, cold_factor=0.1, final_learning_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.warmup_steps = warmup_steps
        self.cold_steps = cold_steps
        self.cold_factor = cold_factor
        self.final_learning_rate = final_learning_rate

    @tf.function
    def __call__(self, step):
        return tf.where(step <= self.cold_steps + self.warmup_steps, 
                        tf.where(step <= self.cold_steps, 
                                 self.initial_learning_rate*self.cold_factor,
                                 self.initial_learning_rate*(self.cold_factor + tf.cast(step - self.cold_steps, tf.float32)*(1 - self.cold_factor)/tf.cast(self.warmup_steps, tf.float32))), 
                        tf.maximum(super().__call__(step - self.cold_steps - self.warmup_steps), self.final_learning_rate))

    def get_config(self):
        config = super().get_config()
        config.update({'warmup_steps': self.warmup_steps})
        config.update({'cold_steps': self.cold_steps})
        config.update({'cold_factor': self.cold_factor})
        config.update({'final_learning_rate': self.final_learning_rate})
        return config
