import numpy as np
from quantumflow.calculus_utils import integrate, integrate_simpson, laplace

def test_colab_devices():
    import os
    import tensorflow as tf

    has_gpu = False
    has_tpu = False

    has_gpu = (tf.test.gpu_device_name() == '/device:GPU:0')

    try:
        device_name = os.environ['COLAB_TPU_ADDR']
        has_tpu = True
    except KeyError:
        pass

    return has_gpu, has_tpu


def unpack_dataset(N, dataset):
    x, potentials, solutions, E = dataset.values()
    density = np.sum(np.square(solutions)[:, :, :N], axis=-1)
    
    dataset_size, discretization_points, _ = solutions.shape
    h = (max(x) - min(x))/(discretization_points-1)
    
    potential = np.expand_dims(potentials, axis=2)*solutions**2
    P = integrate_simpson(potential, h, axis=1)
    K = E - P

    kinetic_energy = np.sum(K[:, :N], axis=-1)
    
    return x, potentials, solutions, E, density, kinetic_energy, dataset_size, discretization_points, h    


class InputPipeline(object):
    def __init__(self, N, dataset_file, is_training=False):
        import pickle
        self.is_training = is_training

        with open(dataset_file, 'rb') as f:
            self.x, self.potentials, _, self.energies, self.densities, self.kenergies, self.M, self.G, self.h = unpack_dataset(N, pickle.load(f))
        self.derivatives = -self.potentials

    def input_fn(self, params):
        import tensorflow as tf

        dataset_densities = tf.data.Dataset.from_tensor_slices(self.densities.astype(np.float32))
        dataset_kenergies = tf.data.Dataset.from_tensor_slices(self.kenergies.astype(np.float32))
        dataset_derivatives = tf.data.Dataset.from_tensor_slices(self.derivatives.astype(np.float32))

        dataset = tf.data.Dataset.zip((dataset_densities, tf.data.Dataset.zip((dataset_kenergies, dataset_derivatives))))

        if self.is_training:
            dataset = dataset.repeat()

        if params['shuffle']:
            dataset = dataset.shuffle(buffer_size=params['shuffle_buffer_size'], seed=params.get('seed', None))

        dataset = dataset.batch(params['batch_size'], drop_remainder=True)
        return dataset

    def features_shape(self):
        return self.densities.shape

    def targets_shape(self):
        return (self.kenergies.shape, self.derivatives.shape)

    def __str__(self):
        string = ''
        if self.is_training:
            string += 'Train Dataset: '
        else:
            string += 'Dataset: '
            
        string += str(self.densities.shape) + ' ' + str(self.kenergies.shape) + ' ' + str(self.derivatives.shape) + ' ' + str(self.densities.dtype)
        return string


def get_resolver():
    import os
    import tensorflow as tf

    try:
        device_name = os.environ['COLAB_TPU_ADDR']
        TPU_WORKER = 'grpc://' + device_name
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_WORKER)
        tf.config.experimental_connect_to_host(resolver.master())
        tf.tpu.experimental.initialize_tpu_system(resolver)

    except KeyError:
        resolver = None

    return resolver

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)/ float(N))
    return cumsum[N:] - cumsum[:-N]


def load_hyperparameters(file_hyperparams, run_name='default', globals=None):
    from ruamel.yaml import YAML

    if globals is not None:
        with open(file_hyperparams) as f:
            globals_list = YAML().load(f)['globals']

    with open(file_hyperparams) as f:
        hparams = YAML().load(f)[run_name]

    if globals is None:
        return hparams

    dicts = [hparams]
    while len(dicts) > 0:
        data = dicts[0]
        for idx, obj in enumerate(data):
            if isinstance(data[obj], dict):
                dicts.append(data[obj])
                continue

            if data[obj] in globals_list:
                data[obj] = globals[data[obj]]
        del dicts[0]
    return hparams


import ipywidgets as widgets
from IPython.display import Audio, HTML, display
from matplotlib import animation, rc
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

def anim_plot(array, x=None, interval=100, bar="", figsize=(15, 3), **kwargs):
    frames = len(array)
    
    if not bar == "":
        import ipywidgets as widgets
        widget = widgets.IntProgress(min=0, max=frames, description=bar, bar_style='success',
                                     layout=widgets.Layout(width='92%'))
        display(widget)

    fig, ax = plt.subplots(figsize=figsize)
    
    if x is None:
        plt_h = ax.plot(array[0], **kwargs)
    else:
        plt_h = ax.plot(x, array[0], **kwargs) 
        
    min_last = np.min(array[-1])
    max_last = np.max(array[-1])
    span_last = max_last - min_last
        
    ax.set_ylim([min_last - span_last*0.2, max_last + span_last*0.2])

    def init():
        return plt_h

    def animate(f):
        if not bar == "":
            widget.value = f

        for i, h in enumerate(plt_h):
            if x is None:
                h.set_data(np.arange(len(array[f][:, i])), array[f][:, i], **kwargs)
            else:
                h.set_data(x, array[f][:, i], **kwargs)
        return plt_h

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=interval,
                                   blit=True, repeat=False)

    plt.close(fig)
    rc('animation', html='html5')
    display(HTML(anim.to_html5_video(embed_limit=1024)))

    if not bar == "":
        widget.close()