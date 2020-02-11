import os
import tensorflow as tf
import numpy as np

def integrate(y, h):
    return h*tf.reduce_sum((y[:, :-1] + y[:, 1:])/2., axis=1, name='trapezoidal_integral_approx')

def laplace(data, h):  # time_axis=1
    temp_laplace = 1 / h ** 2 * (data[:, :-2, :] + data[:, 2:, :] - 2 * data[:, 1:-1, :])
    return tf.pad(temp_laplace, ((0, 0), (1, 1), (0, 0)), 'constant')


def derivative_five_point(density, h):
    return tf.concat([1/(2*h)*(density[:, 2:3] - density[:, 0:1]), 
                      1/(12*h)*(-density[:, 4:] + 8*density[:, 3:-1] - 8*density[:, 1:-3] + density[:, 0:-4]),
                      1/(2*h)*(density[:, -1:] - density[:, -3:-2])], axis=1)

def laplace_five_point(density, h):
    return 1/(12*h**2)*(-density[:, 4:] + 16*density[:, 3:-1] - 30*density[:, 2:-2] + 16*density[:, 1:-3] - density[:, 0:-4])


def weizsaecker_functional(density, h):
    derivative_density = derivative_five_point(density, h)
    inverse_density = 1/density[:, 1:-1]

    weizsaecker_kinetic_energy_density = wked = 1/8*derivative_density**2*inverse_density
    weizsaecker_kinetic_energy_density = tf.concat([2*wked[:, 0:1] - wked[:, 1:2], wked, 2*wked[:, -1:] - wked[:, -2:-1]], axis=1)

    return integrate(weizsaecker_kinetic_energy_density, h)

def weizsaecker_functional_derivative(density, h):
    derivative_density = derivative_five_point(density, h)[:, 1:-1]
    laplace_density = laplace_five_point(density, h)
    inverse_density = 1/density[:, 2:-2]

    weizsaecker_kinetic_energy_functional_derivative = wkefd = 1/8*(derivative_density*inverse_density)**2 - 1/4*laplace_density*inverse_density
    weizsaecker_kinetic_energy_functional_derivative = tf.concat([3*wkefd[:, 0:1] - 2*wkefd[:, 1:2], 2*wkefd[:, 0:1] - wkefd[:, 1:2], wkefd, 2*wkefd[:, -1:] - wkefd[:, -2:-1], 3*wkefd[:, -1:] - 2*wkefd[:, -2:-1]], axis=1)

    return weizsaecker_kinetic_energy_functional_derivative


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
from IPython.display import HTML, display
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


def calculate_density_and_energies(potential, wavefunctions, energies, N, h):
    assert(N <= wavefunctions.shape[2])
    density = np.sum(np.square(wavefunctions)[:, :, :N], axis=-1)

    lpwf = 1/(12*h**2)*(-wavefunctions[:, 4:] + 16*wavefunctions[:, 3:-1] - 30*wavefunctions[:, 2:-2] + 16*wavefunctions[:, 1:-3] - wavefunctions[:, 0:-4])
    laplace_wavefunctions = tf.concat([3*lpwf[:, 0:1] - 2*lpwf[:, 1:2], 2*lpwf[:, 0:1] - lpwf[:, 1:2], lpwf, 2*lpwf[:, -1:] - lpwf[:, -2:-1], 3*lpwf[:, -1:] - 2*lpwf[:, -2:-1]], axis=1) 

    kinetic_energy_densities = -0.5*wavefunctions*laplace_wavefunctions
    potential_energy_densities = np.expand_dims(potential, axis=2)*wavefunctions**2
    
    potential_energies = h * (np.sum(potential_energy_densities, axis=1) - 0.5 * (np.take(potential_energy_densities, 0, axis=1) + np.take(potential_energy_densities, -1, axis=1)))
    kinetic_energies = h * (np.sum(kinetic_energy_densities, axis=1) - 0.5 * (np.take(kinetic_energy_densities, 0, axis=1) + np.take(kinetic_energy_densities, -1, axis=1)))

    energy = np.sum(energies[:, :N], axis=-1)
    potential_energy = np.sum(potential_energies[:, :N], axis=-1)
    kinetic_energy = np.sum(kinetic_energies[:, :N], axis=-1)
    
    kinetic_energy_density = np.sum(kinetic_energy_densities[:, :, :N], axis=-1)
    potential_energy_density = np.sum(potential_energy_densities[:, :, :N], axis=-1)

    return density, energy, potential_energy, kinetic_energy, potential_energy_density, kinetic_energy_density


def calculate_system_properties(potential, wavefunctions, energies, N, h):
    assert(N <= wavefunctions.shape[2])

    density, energy, potential_energy, kinetic_energy, potential_energy_density, kinetic_energy_density = calculate_density_and_energies(potential, wavefunctions, energies, N, h)
    derivative = -potential

    vW_kinetic_energy = weizsaecker_functional(density, h).numpy()
    vW_derivative = weizsaecker_functional_derivative(density, h).numpy()

    return density, energy, potential_energy, kinetic_energy, potential_energy_density, kinetic_energy_density, derivative, vW_kinetic_energy, vW_derivative


class QFDataset():
    def __init__(self, dataset_file, params):
        extension = dataset_file.split('.')[-1]
        
        if extension in ['pkl', 'pickle']:
            import pickle
            with open(dataset_file, 'rb') as f:
                x, h, potential, wavefunctions, energies = pickle.load(f).values()

        elif extension in ['hdf5', 'h5']:
            import h5py
            with h5py.File(dataset_file, 'r') as f:
                x = f.attrs['x']
                h = f.attrs['h']
                potential = f['potential'][()]
                wavefunctions = f['wavefunctions'][()]
                energies = f['energies'][()]
        else:
            raise NotImplementedError('File extension missing or not supported.')  

        if params['N'] == 'all':
            all_data = [calculate_system_properties(potential, wavefunctions, energies, N, h) for N in range(1, energies.shape[1]+1)]
            density, energy, potential_energy, kinetic_energy, potential_energy_density, kinetic_energy_density, derivative, vW_kinetic_energy, vW_derivative = \
                [np.concatenate([all_data[i][j] for i in range(len(all_data))], axis=0) for j in range(len(all_data[0]))]
        else:
            density, energy, potential_energy, kinetic_energy, potential_energy_density, kinetic_energy_density, derivative, vW_kinetic_energy, vW_derivative = \
                calculate_system_properties(potential, wavefunctions, energies, params['N'], h)
            
        self.dataset_size, self.discretisation_points = density.shape

        if params.get('subtract_von_weizsaecker', False):
            kinetic_energy -= params.get('von_weizsaecker_factor', 1.0)*vW_kinetic_energy
            derivative -= params.get('von_weizsaecker_factor', 1.0)*vW_derivative
            
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
        self.potential_energy = potential_energy.astype(self.dtype)
        self.kinetic_energy = kinetic_energy.astype(self.dtype)
        self.potential_energy_density = potential_energy_density.astype(self.dtype)
        self.kinetic_energy_density = kinetic_energy_density.astype(self.dtype)
        self.derivative = derivative.astype(self.dtype)
        self.vW_kinetic_energy = vW_kinetic_energy.astype(self.dtype)
        self.vW_derivative = vW_derivative.astype(self.dtype)

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
            elif name == 'kinetic_energy_density':
                dictionary['kinetic_energy_density'] = self.kinetic_energy_density
            else:
                raise KeyError('feature/target {} does not exist or is not implemented.'.format(name))

        for feature in params['features']:
            add_by_name(self.features, feature)

        for target in params['targets']:
            add_by_name(self.targets, target)

    def get_params(self, shapes=True, h=True, mean=False):
        import numpy as np

        params = {}
        if h:
            params['h'] = self.h

        if shapes:
            params['features_shape'] = {name:feature.shape[1:] for name, feature in self.features.items()}
            params['targets_shape'] = {name:target.shape[1:] for name, target in self.targets.items()}

        if mean:
            params['features_mean'] = {name:np.mean(feature, axis=0) for name, feature in self.features.items()}
            params['targets_mean'] = {name:np.mean(target, axis=0) for name, target in self.targets.items()}

        return params

def run_experiment(experiment, run_name, data_dir='../data'): 
    base_dir = os.path.join(data_dir, experiment)
    model_dir = os.path.join(base_dir, run_name)

    file_model = os.path.join(base_dir, "model.py")
    exec(open(file_model).read(), globals())

    file_hyperparams = os.path.join(base_dir, "hyperparams.config")
    params = load_hyperparameters(file_hyperparams, run_name=run_name, globals=globals())

    train(params, model_dir, data_dir)

def run_multiple(experiment, run_name, data_dir='../data'): 
    import copy

    base_dir = os.path.join(data_dir, experiment)
    model_dir = os.path.join(base_dir, run_name)

    file_model = os.path.join(base_dir, "model.py")
    exec(open(file_model).read(), globals())

    file_hyperparams = os.path.join(base_dir, "hyperparams.config")
    params = load_hyperparameters(file_hyperparams, run_name=run_name, globals=globals())

    def apply_configuration(hparams, configuration):
        dicts = [hparams]
        while len(dicts) > 0:
            data = dicts[0]
            for idx, obj in enumerate(data):
                if obj in ['int_min', 'int_max']:
                    continue

                if isinstance(data[obj], dict):
                    dicts.append(data[obj])
                    continue

                if obj in configuration.keys():
                    data[obj] = configuration[obj]
            del dicts[0]
        return hparams

    def extend_configurations(configurations_out, run_appendices_out, configurations_in, run_appendices_in):
        if len(configurations_out) == 0:
            return configurations_in, run_appendices_in

        merged_configurations = []
        merged_run_appendices = []
        for configuration_out, run_appendix_out in zip(configurations_out, run_appendices_out):
            for configuration_in, run_appendix_in in zip(configurations_in, run_appendices_in):
                merged_configurations.append(configuration_out.copy().update(configuration_in))
                merged_run_appendices.append(run_appendix_out + '_' + run_appendix_in)

        return merged_configurations, merged_run_appendices

    configurations = []
    run_appendices = []

    if 'int_min' in params and 'int_max' in params:
        for (int_key, int_min), (int_key_max, int_max) in zip(params['int_min'].items(), params['int_max'].items()):
            assert int_key == int_key_max 
            int_configurations = []
            int_run_appendices = []
            for int_value in range(int_min, int_max+1):
                int_configurations.append({int_key: int_value})
                int_run_appendices.append(('{}{:0' + str(len(str(int_max))) + 'd}').format(int_key, int_value)) # TODO: support negative values
            configurations, run_appendices = extend_configurations(configurations, run_appendices, int_configurations, int_run_appendices)
    
    for configuration, run_appendix in zip(configurations, run_appendices):
        train(apply_configuration(copy.deepcopy(params), configuration), os.path.join(model_dir, run_appendix), data_dir)


def build_model(params, data_dir='../data', dataset_train=None):
    if dataset_train is None:
        dataset_train = QFDataset(os.path.join(data_dir, params['dataset_train']), params)
        params['dataset'] = dataset_train.get_params(shapes=True, h=True, mean=True)
    
    tf.keras.backend.clear_session()
    return params['model'](params)

def train(params, model_dir=None, data_dir='../data', callbacks=None):
    dataset_train = QFDataset(os.path.join(data_dir, params['dataset_train']), params)
    dataset_validate = QFDataset(os.path.join(data_dir, params['dataset_validate']), params) if 'dataset_validate' in params else None
    params['dataset'] = dataset_train.get_params(shapes=True, h=True, mean=True)

    tf.keras.backend.clear_session()
    if 'seed' in params:
        tf.random.set_seed(params['seed'])
        
    model = build_model(params, data_dir=data_dir, dataset_train=dataset_train)

    optimizer_kwargs = params['optimizer_kwargs'].copy()
    if isinstance(params['optimizer_kwargs']['learning_rate'], float):
        learning_rate = params['optimizer_kwargs']['learning_rate']
    elif isinstance(params['optimizer_kwargs']['learning_rate'], str):
        optimizer_kwargs['learning_rate'] = learning_rate = getattr(tf.keras.optimizers.schedules, params['optimizer_kwargs']['learning_rate'])(**params['optimizer_kwargs']['learning_rate_kwargs'])
        del optimizer_kwargs['learning_rate_kwargs']
    elif issubclass(params['optimizer_kwargs']['learning_rate'], tf.keras.optimizers.schedules.LearningRateSchedule):
        optimizer_kwargs['learning_rate'] = learning_rate = params['optimizer_kwargs']['learning_rate'](**params['optimizer_kwargs']['learning_rate_kwargs'])
        del optimizer_kwargs['learning_rate_kwargs']

    optimizer = getattr(tf.keras.optimizers, params['optimizer'])(**optimizer_kwargs) if isinstance(params['optimizer'], str) else params['optimizer'](**optimizer_kwargs)
    model.compile(optimizer, loss=params['loss'], loss_weights=params.get('loss_weights', None), metrics=params.get('metrics', None))

    if params.get('load_checkpoint', None) is not None:
        model.load_weights(os.path.join(data_dir, params['load_checkpoint']))
        if params['fit_kwargs'].get('verbose', 0) > 0:
            print("loading weights from ", os.path.join(data_dir, params['load_checkpoint']))

    if callbacks is None:
        callbacks = []

    if model_dir is not None and params.get('checkpoint', False):
        checkpoint_params = params['checkpoint_kwargs'].copy()
        checkpoint_params['filepath'] = os.path.join(model_dir, checkpoint_params.pop('filename', 'weights.{epoch:05d}.hdf5'))
        checkpoint_params['verbose'] = checkpoint_params.get('verbose', min(1, params['fit_kwargs'].get('verbose', 1)))
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(**checkpoint_params))

    if model_dir is not None and params.get('tensorboard', False):
        tensorboard_callback_class = params['tensorboard'] if callable(params['tensorboard']) else tf.keras.callbacks.TensorBoard
        callbacks.append(tensorboard_callback_class(log_dir=model_dir, learning_rate=learning_rate, **params['tensorboard_kwargs']))

    model.fit(x=dataset_train.features, 
              y=dataset_train.targets, 
              callbacks=callbacks,
              validation_data=(dataset_validate.features, dataset_validate.targets) if dataset_validate is not None else None,
              **params['fit_kwargs'])

    if model_dir is not None and params['save_model'] is True:
        model.save(os.path.join(model_dir, 'model.h5')) 

    if model_dir is not None and params['export'] is True:
        export_model = getattr(model, params['export_model']) if not params.get('export_model', 'self') == 'self' else model
        tf.saved_model.save(export_model, os.path.join(model_dir, 'saved_model'))

    return model, params
