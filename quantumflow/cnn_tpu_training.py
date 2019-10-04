
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import summary
from tensorflow.contrib.training.python.training import evaluation

def conv_nn(input, return_layers=False, filters=(16, 16, 16), kernel_size=(121, 121, 121), strides=(1, 1, 1), padding='valid', activation=tf.nn.softplus, **kwargs):
    layers_list = []
    value = tf.expand_dims(input, axis=-1)
    value = tf.expand_dims(value, axis=2)
    layers_list.append(tf.reduce_sum(value, axis=2))
    
    assert len(filters) == len(kernel_size)
    layers = len(filters)
    
    for l in range(layers -1):
        value = tf.layers.conv2d(value, filters=filters[l], kernel_size=(kernel_size[l], 1), strides=strides[l], activation=activation, padding=padding, **kwargs)
        layers_list.append(tf.reduce_sum(value, axis=2))
        
    value = tf.layers.conv2d(value, filters=filters[-1], kernel_size=(kernel_size[-1], 1), padding=padding, **kwargs)
    value = tf.reduce_sum(value, axis=2)
    layers_list.append(value)
    
    value = tf.reduce_sum(value, axis=2)
    value = tf.reduce_sum(value, axis=1)
    layers_list.append(value)
    
    if return_layers:
        return value, layers_list
    else:
        return value


class SineWaveInitializer(tf.initializers.variance_scaling):
    def __call__(self, shape, dtype=None, partition_info=None):
        G = shape[0]
        shape[0] = 1

        weights = super().__call__(shape=shape, dtype=dtype, partition_info=partition_info)

        lin = tf.reshape(tf.linspace(0.0, np.pi, G), (G, 1, 1, 1))
        freq = tf.reshape(tf.range(shape[-1], dtype=tf.float32)+2, (1, 1, 1, shape[-1]))
        sine = tf.sin(lin*freq)/G

        return sine*weights


def learning_rate_schedule(params, global_step):
    batches_per_epoch = params['train_total_size'] / params['train_batch_size']
    current_epoch = tf.cast((tf.cast(global_step, tf.float32) / batches_per_epoch), tf.int32)

    initial_learning_rate = params['learning_rate']

    if params['use_learning_rate_warmup']:
        warmup_decay = params['learning_rate_decay']**(
        (params['warmup_epochs'] + params['cold_epochs']) /
        params['learning_rate_decay_epochs'])
        adj_initial_learning_rate = initial_learning_rate * warmup_decay

    final_learning_rate = params['final_learning_rate_factor'] * initial_learning_rate

    learning_rate = tf.train.exponential_decay(
        learning_rate=initial_learning_rate,
        global_step=global_step,
        decay_steps=int(params['learning_rate_decay_epochs'] * batches_per_epoch),
        decay_rate=params['learning_rate_decay'],
        staircase=True)

    if params['use_learning_rate_warmup']:
        wlr = 0.1 * adj_initial_learning_rate
        wlr_height = tf.cast(0.9 * adj_initial_learning_rate / 
                                (params['warmup_epochs'] + params['learning_rate_decay_epochs'] - 1), tf.float32)
        
        epoch_offset = tf.cast(params['cold_epochs'] - 1, tf.int32)
        exp_decay_start = (params['warmup_epochs'] + params['cold_epochs'] + params['learning_rate_decay_epochs'])

        lin_inc_lr = tf.add(wlr, tf.multiply(tf.cast(tf.subtract(current_epoch, epoch_offset), tf.float32), wlr_height))

        learning_rate = tf.where(
            tf.greater_equal(current_epoch, params['cold_epochs']),
            (tf.where(tf.greater_equal(current_epoch, exp_decay_start), learning_rate, lin_inc_lr)), 
            tf.ones_like(learning_rate)*wlr)

    # Set a minimum boundary for the learning rate.
    learning_rate = tf.maximum(learning_rate, final_learning_rate, name='learning_rate')

    return learning_rate


def deriv_conv_nn_model_fn(features, labels, mode, params):

    if isinstance(features, dict):
        features = features['feature']

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    is_eval = (mode == tf.estimator.ModeKeys.EVAL)   

    target_prediction = conv_nn(features, **params['kwargs'])
    derivative_prediction = 1/params['h']*tf.gradients(target_prediction, features)[0]

    predictions = {
        'value': target_prediction,
        'derivative': derivative_prediction
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
            'regression': tf.estimator.export.PredictOutput(predictions)
            })
    
    target, derivative = labels
    
    loss_y = tf.losses.mean_squared_error(target_prediction, target)
    loss_gradient = tf.losses.mean_squared_error(derivative_prediction, derivative)

    loss_l2 = []
    for v in tf.trainable_variables():
        if 'kernel' in v.name:
            loss_l2.append(tf.nn.l2_loss(v))
    loss_l2 = tf.add_n(loss_l2)
    
    loss = loss_y + params['balance']*loss_gradient
    
    if params['l2_loss'] > 0.0:
        loss += params['l2_loss']*loss_l2

    host_call = None
    train_op = None

    if is_training:
        batches_per_epoch = params['train_total_size'] / params['train_batch_size']
        global_step = tf.train.get_or_create_global_step()
        current_epoch = tf.cast((tf.cast(global_step, tf.float32) / batches_per_epoch), tf.int32)
        learning_rate = learning_rate_schedule(params, global_step)
        #tf.summary.scalar('lr', learning_rate) # doesn't work on TPU

        if params['optimizer'] == 'Adam':
            print('Using Adam optimizer')
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif params['optimizer'] == 'sgd':
            print('Using SGD optimizer')
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif params['optimizer'] == 'momentum':
            print('Using Momentum optimizer')
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        elif params['optimizer'] == 'RMS':
            print('Using RMS optimizer')
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
        else:
            tf.logging.fatal('Unknown optimizer:', params['optimizer'])

        if params['gradient_clipping']:
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, params['gradient_clip_norm'])

        if params['use_tpu']:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)

        # To log the loss, current learning rate, and epoch for Tensorboard, the
        # summary op needs to be run on the host CPU via host_call. host_call
        # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
        # dimension. These Tensors are implicitly concatenated to
        # [params['batch_size']].
        gs_t = tf.reshape(global_step, [1])
        #loss_t = tf.reshape(loss, [1])
        #loss_y_t = tf.reshape(loss_y, [1])
        #loss_gradient_t = tf.reshape(loss_gradient, [1])
        #loss_l2_t = tf.reshape(loss_l2, [1])
        lr_t = tf.reshape(learning_rate, [1])
        ce_t = tf.reshape(current_epoch, [1])

        if not params['skip_host_call']:
            def host_call_fn(gs, lr, ce):
                gs = gs[0]
                with summary.create_file_writer(params['model_dir']).as_default():
                    with summary.always_record_summaries():
                        #summary.scalar('loss', tf.reduce_mean(loss), step=gs)
                        #summary.scalar('loss_y', tf.reduce_mean(loss_y), step=gs)
                        #summary.scalar('loss_gradient', tf.reduce_mean(loss_gradient), step=gs)
                        #summary.scalar('loss_l2', tf.reduce_mean(loss_l2), step=gs)

                        summary.scalar('learning_rate', tf.reduce_mean(lr), step=gs)
                        summary.scalar('current_epoch', tf.reduce_mean(ce), step=gs)

                        return summary.all_summary_ops()

            host_call = (host_call_fn, [gs_t, lr_t, ce_t])

    eval_metrics = None
    if is_eval:
        def metric_fn(target_prediction, target, derivative_prediction, derivative):
            return {
                'value_mae': tf.metrics.mean_absolute_error(target_prediction, target),
                'derivative_mae': tf.metrics.mean_absolute_error(derivative_prediction, derivative),
            }

        eval_metrics = (metric_fn, [target_prediction, target, derivative_prediction, derivative])

    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        host_call=host_call,
        eval_metrics=eval_metrics)


def train(params, resolver, dataset_train, dataset_eval):
    tpu_config = tf.contrib.tpu.TPUConfig(iterations_per_loop=params['iterations'], num_shards=params['num_shards'])

    run_config = tf.contrib.tpu.RunConfig(
        cluster=resolver,
        model_dir=params['model_dir'],
        tf_random_seed=params['seed'],
        save_checkpoints_secs=params['save_checkpoints_secs'],
        keep_checkpoint_max=params['keep_checkpoint_max'],
        save_summary_steps=params['save_summary_steps'],
        session_config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=params['log_device_placement']),
        tpu_config=tpu_config)

    model = tf.contrib.tpu.TPUEstimator(
        model_fn=params['model_fn'],
        use_tpu=params['use_tpu'],
        config=run_config,
        params=params,
        train_batch_size=params['train_batch_size'],
        eval_batch_size=params['eval_batch_size'])

    print('Training for {} steps with batch size {}, returning to CPU every {} steps\n'
        'summary every {} steps, saving every {} seconds.'.format(params['train_steps'], params['train_batch_size'], params['iterations'], 
                                                                    params['save_summary_steps'], params['save_checkpoints_secs']))

    latest_checkpoint = model.latest_checkpoint()
    current_step = int(latest_checkpoint.split('-')[-1]) if latest_checkpoint is not None else 0
    while current_step < params['train_steps']:
        train_steps = params['train_steps_per_eval'] if current_step % params['train_steps_per_eval'] == 0 else \
                                                        params['train_steps_per_eval'] - current_step % params['train_steps_per_eval']
        cycle = current_step // params['train_steps_per_eval']
        print('Starting training cycle {} - training for {} steps.'.format(cycle, train_steps))
        model.train(input_fn=dataset_train.input_fn, steps=train_steps)
        current_step += train_steps

        print('Starting evaluation cycle {}.'.format(cycle))
        eval_results = model.evaluate(input_fn=dataset_eval.input_fn, steps=params['eval_total_size'] // params['eval_batch_size'])
        print('Evaluation results: {}'.format(eval_results))

    def serving_input_receiver_fn():
        features = tf.placeholder(dtype=tf.float32, shape=[None] + list(dataset_eval.features_shape()[1:]))
        receiver_tensors = {'features': features}
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    export_path = os.path.join(params['model_dir'], 'saved_model')
    print("Exporting model to {} with input placeholder {}".format(export_path, [None] + list(dataset_eval.features_shape()[1:])))
    model.export_saved_model(export_path, serving_input_receiver_fn)