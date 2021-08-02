import os
import sys
import argparse

import tensorflow as tf
import quantumflow


def main(experiment, run_name):
    project_path = os.path.expanduser('~/QuantumFlow')
    base_dir = os.path.join(project_path, "experiments", experiment)
    params = quantumflow.utils.load_yaml(os.path.join(base_dir, 'hyperparams.yaml'))[run_name]
    run_dir = os.path.join(base_dir, run_name)

    if not os.path.exists(run_dir): os.makedirs(run_dir)

    dataset_train = quantumflow.instantiate(params['dataset_train'], run_dir=run_dir)
    dataset_train.build()

    dataset_validate = quantumflow.instantiate(params['dataset_validate'], run_dir=run_dir)
    dataset_validate.build()
    
    tf.keras.backend.clear_session()
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

    if params.get('load_checkpoint', None) is not None:
        model.load_weights(os.path.join(data_dir, params['load_checkpoint']))
        if params['fit'].get('verbose', 0) > 0:
            print("loading weights from ", os.path.join(data_dir, params['load_checkpoint']))

    callbacks = []

    if run_dir is not None and params.get('checkpoint', False):
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


    if params['save_model'] is True:
        model.save(os.path.join(run_dir, 'model.h5')) 

    if params['export'] is True:
        export_model = getattr(model, params['export_model']) if not params.get('export_model', 'self') == 'self' else model
        tf.saved_model.save(export_model, os.path.join(run_dir, 'saved_model'))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Dataset script.')
    parser.add_argument('experiment', type=str, help='experiment')
    parser.add_argument('run_name', type=str, help='name of the dataset')
    
    args = parser.parse_args()
    main(args.experiment, args.run_name)
