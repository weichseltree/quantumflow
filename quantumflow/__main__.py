import os
import argparse


def build_dataset(dataset, run_name, force=False):
    import quantumflow
    run_dir, params = quantumflow.get_dataset_dir_and_params(dataset, run_name)
    dataset = quantumflow.instantiate(params, run_dir=run_dir)
    dataset.build(force=force)
    return dataset
    
    
def train_model(experiment, run_name):
    import quantumflow
    import tensorflow as tf
    
    run_dir, params = quantumflow.get_experiment_dir_and_params(experiment, run_name)
    quantumflow.ensure_dir(run_dir)

    dataset_train = quantumflow.instantiate(params['dataset_train'], run_dir=run_dir).build()
    dataset_validate = quantumflow.instantiate(params['dataset_validate'], run_dir=run_dir).build()
    
    tf.profiler.experimental.server.start(6009)
    tf.keras.backend.clear_session()
    
    if 'seed' in params:
        tf.random.set_seed(params['seed'])

    model = quantumflow.instantiate(params['model'], run_dir=run_dir, dataset=dataset_train)
    print(model.summary())
    
    optimizer = quantumflow.instantiate(params['optimizer'])

    model.compile(
        optimizer,
        loss=params['loss'],
        loss_weights=params.get('loss_weights', None),
        metrics=params.get('metrics', None)
    )

    if params.get('load_weights', None) is not None:
        model.load_weights(os.path.join(run_dir, params['load_weights']))
        if params['fit'].get('verbose', 0) > 0:
            print("loading weights from ", os.path.join(run_dir, params['load_checkpoint']))

    callbacks = []

    if params.get('checkpoint', False):
        checkpoint_params = params['checkpoint'].copy()
        checkpoint_params['filepath'] = os.path.join(run_dir, checkpoint_params.pop('filename', 'weights.{epoch:05d}.hdf5'))
        checkpoint_params['verbose'] = checkpoint_params.get('verbose', min(1, params['fit'].get('verbose', 1)))
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(**checkpoint_params))

    if 'tensorboard' in params:
        callbacks.append(
            quantumflow.instantiate(params['tensorboard'], log_dir=run_dir, learning_rate=optimizer.learning_rate))

    model.fit(x=dataset_train.features,
              y=dataset_train.targets,
              callbacks=callbacks,
              validation_data=(dataset_validate.features, dataset_validate.targets) if dataset_validate is not None else None,
              **params['fit'])

    if params.get('save', True):
        save_target = getattr(model, params['save_target']) if not params.get('save_target', 'self') == 'self' else model
        save_target.save(os.path.join(run_dir, 'saved_model'), include_optimizer=False)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QuantumFlow')
    parser.add_argument('command', type=str, help='One of [build_dataset, train_model]')
    parser.add_argument('experiment_dataset', type=str, help='experiment or dataset name')
    parser.add_argument('run_name', type=str, help='YAML entry: run name')
    args = parser.parse_args()
    
    if args.command == 'build_dataset':
        build_dataset(args.experiment_dataset, args.run_name, force=True)
    elif args.command == 'train_model':
        train_model(args.experiment_dataset, args.run_name)

