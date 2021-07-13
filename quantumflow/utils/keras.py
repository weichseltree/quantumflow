import tensorflow as tf
import time

import quantumflow

class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, *args, metrics_freq=0, learning_rate=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_time = None
        self.last_step = None
        self.learning_rate = learning_rate
        self.metrics_freq = metrics_freq


'''
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
'''

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

def Adam(learning_rate, **kwargs):
    if not isinstance(learning_rate, float):
        learning_rate = quantumflow.instantiate(learning_rate)

    return tf.keras.optimizers.Adam(learning_rate=learning_rate, **kwargs)
