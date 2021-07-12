import quantumflow

import os

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
        params['dataset'] = dataset_train.get_params(**params['dataset_info'])
    
    tf.keras.backend.clear_session()
    return params['model'](params)

def train(params, model_dir=None, data_dir='../data', callbacks=None):
    dataset_train = QFDataset(os.path.join(data_dir, params['dataset_train']), params)
    dataset_validate = QFDataset(os.path.join(data_dir, params['dataset_validate']), params) if 'dataset_validate' in params else None
    params['dataset'] = dataset_train.get_params(**params['dataset_info'])

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