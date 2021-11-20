import tensorflow as tf


class Dataset():
    def __init__(self, run_dir, **kwargs):
        self.run_dir = run_dir

    def build(self, force=False):
        return self

    def visualize(self):
        pass

    def iter(self):
        raise NotImplementedError()


class Model(tf.keras.Model):
    
    @property
    def debug(self):
        return False
    
    @debug.setter
    def debug(self, value):
        for layer in self.layers:
            if hasattr(layer, 'debug'):
                layer.debug = True
