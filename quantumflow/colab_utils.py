import numpy as np

def integrate(y, h):
    return h*tf.reduce_sum((y[:, :-1] + y[:, 1:])/2., axis=1, name='trapezoidal_integral_approx')

def laplace(data, h):  # time_axis=1
    temp_laplace = 1 / h ** 2 * (data[:, :-2, :] + data[:, 2:, :] - 2 * data[:, 1:-1, :])
    return tf.pad(temp_laplace, ((0, 0), (1, 1), (0, 0)), 'constant')


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

class QFDataset():
    def __init__(self, dataset_file, params, set_h=False, set_shapes=False, set_mean=False):
        import pickle
        import numpy as np

        with open(dataset_file, 'rb') as f:
            x, h, potential, wavefunctions, energies = pickle.load(f).values()
            self.dataset_size, self.discretisation_points, self.max_N = wavefunctions.shape
            assert(params['N'] <= self.max_N)
            density = np.sum(np.square(wavefunctions)[:, :, :params['N']], axis=-1)

            potential_energy_densities = np.expand_dims(potential, axis=2)*wavefunctions**2
            potential_energies = h * (np.sum(potential_energy_densities, axis=1) - 0.5 * (np.take(potential_energy_densities, 0, axis=1) + np.take(potential_energy_densities, -1, axis=1)))
            kinetic_energies = energies - potential_energies

            energy = np.sum(energies[:, :params['N']], axis=-1)
            kinetic_energy = np.sum(kinetic_energies[:, :params['N']], axis=-1)


        if params['dtype'] == 'double' or params['dtype'] == 'float64':
            if potential.dtype == np.float32:
                raise ImportError("requested dtype={}, but dataset is saved with dtype={}, which is less precise.".format(params['dtype'], potential.dtype))
            self.dtype = np.float64
        elif params['dtype'] == 'float' or params['dtype'] == 'float32':
            self.dtype = np.float32
        else:
            raise ValueError('unknown dtype {}'.format(params['dtype']))

        self.x = x.astype(self.dtype)
        self.h = h.astype(self.dtype)
        self.potential = potential.astype(self.dtype)
        self.density = density.astype(self.dtype)
        self.energy = energy.astype(self.dtype)
        self.kinetic_energy = kinetic_energy.astype(self.dtype)
        self.derivative = -self.potential

        if not 'features' in params or not 'targets' in params: 
            return

        self.features = {}
        self.targets = {}

        def add_by_name(dictionary, name):
            if name == 'density':
                dictionary['density'] = self.density
            elif name == 'derivative':
                dictionary['derivative'] = self.derivative
            elif name == 'potential':
                dictionary['potential'] = self.potential
            elif name == 'kinetic_energy':
                dictionary['kinetic_energy'] = self.kinetic_energy
            else:
                raise KeyError('feature/target {} does not exist or is not implemented.'.format(name))

        for feature in params['features']:
            add_by_name(self.features, feature)

        for target in params['targets']:
            add_by_name(self.targets, target)

        if set_h:
            params['h'] = self.h

        if set_shapes:
            params['features_shape'] = {name:feature.shape[1:] for name, feature in self.features.items()}
            params['targets_shape'] = {name:target.shape[1:] for name, target in self.targets.items()}

        if set_mean:
            params['features_mean'] = {name:np.mean(feature, axis=0) for name, feature in self.features.items()}
            params['targets_mean'] = {name:np.mean(target, axis=0) for name, target in self.targets.items()}


    @property
    def dataset(self):
        import tensorflow as tf
        return tf.data.Dataset.zip((tf.data.Dataset.zip({name:tf.data.Dataset.from_tensor_slices(feature) for name, feature in self.features.items()}), 
                                    tf.data.Dataset.zip({name:tf.data.Dataset.from_tensor_slices(target) for name, target in self.targets.items()})))