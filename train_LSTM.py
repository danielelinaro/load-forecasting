import os
import re
import sys
import glob
import json
import pickle
import datetime
import argparse as arg
from time import strftime, localtime

import numpy as np
import matplotlib.pyplot as plt

from comet_ml import Experiment
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, callbacks, models
import tensorflow_addons as tfa
from sklearn.preprocessing import OneHotEncoder

import colorama as cm
print_error   = lambda msg: print(f'{cm.Fore.RED}'    + msg + f'{cm.Style.RESET_ALL}')
print_warning = lambda msg: print(f'{cm.Fore.YELLOW}' + msg + f'{cm.Style.RESET_ALL}')
print_msg     = lambda msg: print(f'{cm.Fore.GREEN}'  + msg + f'{cm.Style.RESET_ALL}')


LEARNING_RATE = []

class LearningRateCallback(keras.callbacks.Callback):
    def __init__(self, model, experiment = None):
        self.model = model
        self.experiment = experiment
        self.step = 0

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        try:
            lr = self.model.optimizer.learning_rate(self.step).numpy()
        except:
            lr = self.model.optimizer.learning_rate
        LEARNING_RATE.append(lr)
        if self.experiment is not None:
            self.experiment.log_metric('learning_rate', lr, self.step)


def make_data_blocks(dataset, start_index, end_index, history_size, target_size,
                     steps_ahead=0, target=None, step=1):
    data = []
    labels = []

    if target is None:
        univariate = True
    else:
        univariate = False

    start_index = start_index + history_size
    if end_index is None or end_index >= len(dataset) - (target_size + steps_ahead) + 1:
        end_index = len(dataset) - (target_size + steps_ahead) + 1

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        if univariate:
            # Reshape data from (history_size,) to (history_size, 1)
            data.append(np.reshape(dataset[indices], (history_size, 1)))
            labels.append(dataset[i + steps_ahead : i + steps_ahead + target_size])
        else:
            data.append(dataset[indices])
            labels.append(target[i + steps_ahead : i + steps_ahead + target_size])
    return np.array(data), np.array(labels)


def build_model(input_shape, model_arch, loss_fun_pars, optimizer_pars, lr_schedule_pars):

    loss_fun_name = loss_fun_pars['name'].lower()
    if loss_fun_name == 'mae':
        loss = losses.MeanAbsoluteError()
    elif loss_fun_name == 'mape':
        loss = losses.MeanAbsolutePercentageError()
    else:
        raise Exception('Unknown loss function: {}.'.format(loss_function))

    if lr_schedule_pars is not None and 'name' in lr_schedule_pars:
        if lr_schedule_pars['name'] == 'cyclical':
            # According to [1], "experiments show that it often is good to set stepsize equal to
            # 2 − 10 times the number of iterations in an epoch".
            #
            # [1] Smith, L.N., 2017, March.
            #     Cyclical learning rates for training neural networks.
            #     In 2017 IEEE Winter Conference on Applications of Computer Vision (WACV) (pp. 464-472). IEEE.
            #
            step_sz = steps_per_epoch * lr_schedule_pars['factor']
            learning_rate = tfa.optimizers.Triangular2CyclicalLearningRate(
                initial_learning_rate = lr_schedule_pars['initial_learning_rate'],
                maximal_learning_rate = lr_schedule_pars['max_learning_rate'],
                step_size = step_sz)
            print_msg(f'Will use cyclical learning rate scheduling with a step size of {step_sz}.')
        elif lr_schedule_pars['name'] == 'exponential_decay':
            initial_learning_rate = lr_schedule_pars['initial_learning_rate']
            decay_steps = lr_schedule_pars['decay_steps']
            decay_rate = lr_schedule_pars['decay_rate']
            learning_rate = optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps, decay_rate)
            print_msg('Will use exponential decay of learning rate.')
        else:
            raise Exception(f'Unknown learning rate schedule: {lr_schedule_pars["name"]}')
    else:
        learning_rate = optimizer_pars['learning_rate']

    optimizer_name = optimizer_pars['name'].lower()
    if optimizer_name == 'sgd':
        momentum = optimizer_pars['momentum'] if 'momentum' in optimizer_pars else 0.
        nesterov = optimizer_pars['nesterov'] if 'nesterov' in optimizer_pars else False
        optimizer = optimizers.SGD(learning_rate, momentum, nesterov)
    elif optimizer_name in ('adam', 'adamax', 'nadam'):
        beta_1 = optimizer_pars['beta_1'] if 'beta_1' in optimizer_pars else 0.9
        beta_2 = optimizer_pars['beta_2'] if 'beta_2' in optimizer_pars else 0.999
        if optimizer_name == 'adam':
            optimizer = optimizers.Adam(learning_rate, beta_1, beta_2)
        elif optimizer_name == 'adamax':
            optimizer = optimizers.Adamax(learning_rate, beta_1, beta_2)
        else:
            optimizer = optimizers.Nadam(learning_rate, beta_1, beta_2)
    elif optimizer_name == 'adagrad':
        initial_accumulator_value = optimizer_pars['initial_accumulator_value'] if \
                                    'initial_accumulator_value' in optimizer_pars else 0.1
        optimizer = optimizers.Adagrad(learning_rate, initial_accumulator_value)
    elif optimizer_name == 'adadelta':
        rho = optimizer_pars['rho'] if 'rho' in optimizer_pars else 0.95
        optimizer = optimizers.Adadelta(learning_rate, rho)
    else:
        raise Exception('Unknown optimizer: {}.'.format(optimizer_name))

    n_layers = model_arch['N_layers']
    n_units = model_arch['N_units']
    if np.isscalar(n_units):
        n_units = [n_units for _ in range(n_layers)]

    inputs = tf.keras.Input(shape=input_shape[-2:], name='input')
    for n in n_units[:-1]:
        try:
            lyr = tf.keras.layers.LSTM(n, return_sequences=True)(lyr)
        except:
            lyr = tf.keras.layers.LSTM(n, return_sequences=True)(inputs)
    try:
        lyr = tf.keras.layers.LSTM(n_units[-1])(lyr)
    except:
        lyr = tf.keras.layers.LSTM(n_units[-1])(inputs)
    outputs = tf.keras.layers.Dense(target_size)(lyr)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss)

    return model, optimizer, loss


def train_model(model, training_dataset, validation_dataset, N_epochs, steps_per_epoch,
                validation_steps, output_dir, experiment, callback_pars, verbose = 1):

    checkpoint_dir = output_dir + '/checkpoints'
    os.makedirs(checkpoint_dir)

    # create a callback that saves the model's weights
    checkpoint_cb = callbacks.ModelCheckpoint(filepath = checkpoint_dir + \
                                              '/weights.{epoch:04d}-{val_loss:.6f}.h5',
                                              save_weights_only = False,
                                              save_best_only = True,
                                              monitor = 'val_loss',
                                              verbose = verbose)
    print_msg('Added callback for saving weights at checkpoint.')

    cbs = [checkpoint_cb, LearningRateCallback(model, experiment)]
    print_msg('Added callback for logging learning rate.')

    try:
        for cb_pars in callbacks_pars:
            if cb_pars['name'] == 'early_stopping':
                # create a callback that will stop the optimization if there is no improvement
                early_stop_cb = callbacks.EarlyStopping(monitor = cb_pars['monitor'],
                                                        patience = cb_pars['patience'],
                                                        verbose = verbose,
                                                        mode = cb_pars['mode'])
                cbs.append(early_stop_cb)
                print_msg('Added callback for early stopping.')
            elif cb_pars['name'] == 'reduce_on_plateau':
                lr_scheduler_cb = callbacks.ReduceLROnPlateau(monitor = cb_pars['monitor'],
                                                              factor = cb_pars['factor'],
                                                              patience = cb_pars['patience'],
                                                              verbose = verbose,
                                                              mode = cb_pars['mode'],
                                                              cooldown = cb_pars['cooldown'],
                                                              min_lr = cb_pars['min_lr'])
                cbs.append(lr_scheduler_cb)
                print_msg('Added callback for reducing learning rate on plateaus.')
            else:
                raise Exception(f'Unknown callback: {cb_pars["name"]}')
    except:
        print_warning('Not adding callbacks.')

    return model.fit(training_dataset,
                     epochs = N_epochs,
                     steps_per_epoch = steps_per_epoch,
                     validation_data = validation_dataset,
                     validation_steps = validation_steps,
                     verbose = verbose,
                     callbacks = cbs)


    
if __name__ == '__main__':

    progname = os.path.basename(sys.argv[0])
    
    parser = arg.ArgumentParser(description = 'Train a network to estimate inertia',
                                formatter_class = arg.ArgumentDefaultsHelpFormatter, \
                                prog = progname)
    parser.add_argument('config_file', type=str, action='store', help='configuration file')
    parser.add_argument('--hours-ahead',  default=None,  type=float, help='hours ahead for prediction (overwrites value in configuration file)')
    parser.add_argument('-o', '--output-dir',  default='experiments',  type=str, help='output directory')
    parser.add_argument('--no-comet', action='store_true', help='do not use CometML to log the experiment')
    args = parser.parse_args(args=sys.argv[1:])

    config_file = args.config_file
    if not os.path.isfile(config_file):
        print_error('{}: {}: no such file.'.format(progname, config_file))
        sys.exit(1)
    config = json.load(open(config_file, 'r'))

    with open('/dev/urandom', 'rb') as fid:
        seed = int.from_bytes(fid.read(4), 'little')
    tf.random.set_seed(seed)
    print_msg('Seed: {}'.format(seed))

    if args.hours_ahead is not None:
        config['hours_ahead'] = args.hours_ahead
        print_msg('Hours ahead for prediction: {:g}.'.format(config['hours_ahead']))

    log_to_comet = not args.no_comet and False
    
    if log_to_comet:
        ### create a CometML experiment
        experiment = Experiment(
            api_key = os.environ['COMET_API_KEY'],
            project_name = 'inertia',
            workspace = 'danielelinaro'
        )
        experiment_key = experiment.get_key()
    else:
        experiment = None
        experiment_key = strftime('%Y%m%d-%H%M%S', localtime())

    time_step = config['time_step']

    data_file = config['data_file']
    if not os.path.isfile(data_file):
        print_error('{}: {}: no such file.'.format(progname, data_file))
        sys.exit(1)

    ### load the data
    full_data = pickle.load(open(data_file, 'rb'))
    building_energy = full_data['full']['building_energy']

    t0 = datetime.datetime.combine(datetime.date.today(),
                                   building_energy['datetime'][0].to_pydatetime().time())
    t1 = datetime.datetime.combine(datetime.date.today(),
                                   building_energy['datetime'][1].to_pydatetime().time())
    orig_time_step = int((t1 - t0).total_seconds() / 60) # [min]
    avg_step = time_step // orig_time_step

    building_energy['consumption_averaged'] = building_energy['consumption'].rolling(window=avg_step).mean()
    building_energy['generation_averaged'] = building_energy['generation'].rolling(window=avg_step).mean()
    building_energy['minute'] = [(dt.hour * 60 + dt.minute) for dt in building_energy['datetime']]
    building_energy['workday'] = np.logical_not(building_energy['weekend'] | building_energy['holiday'])
    cols = building_energy.columns.tolist()
    cols = cols[:1] + cols[-2:] + cols[1:-2]
    building_energy = building_energy[cols]
    building_energy = building_energy[avg_step - 1 : : avg_step]

    samples_per_day = 24 * 60 // time_step
    n_samples = building_energy.shape[0]
    n_days = n_samples // samples_per_day
    t = np.arange(samples_per_day) * time_step / 60
    print(f'Time step: {time_step} minutes.')
    print(f'Number of days: {n_days}.')
    print(f'Samples per day: {samples_per_day}.')

    ### how many days to use for training, test and validation
    n_days_training = int(config['data_split']['training'] * n_days)
    n_days_test = int(config['data_split']['test'] * n_days)
    n_days_validation = n_days - n_days_training - n_days_test
    train_split = n_days_training * samples_per_day
    validation_split = (n_days_training + n_days_validation) * samples_per_day
    print('Will use the first {} measurements (corresponding to {} days) to train the network.'.\
          format(train_split, n_days_training))
    print('Will use the subsequent {} measurements (corresponding to {} days) to validate the network.'.\
          format(validation_split - train_split, n_days_validation))
    print('Will use the final {} measurements (corresponding to {} days) to test the network.'.\
          format(n_days_test * samples_per_day, n_days_test))

    ### normalize the data
    cols = config['inputs']['continuous']
    if config['average_continuous_inputs']:
        cols = [col + '_averaged' for col in cols]
    building_data = building_energy[cols].to_numpy(dtype=np.float32)
    building_data = np.reshape(building_data, [n_days, samples_per_day, building_data.shape[1]])
    ##### change here when more data will be used in the training
    data = building_data
    training_set_max = np.max(data[:train_split, :, :], axis=(0, 1))
    training_set_min = np.min(data[:train_split, :, :], axis=(0, 1))
    data_scaled = np.array([np.ndarray.flatten(-1 + 2 * (data[:,:,i] - m) / (M - m))
                            for i,(M,m) in enumerate(zip(training_set_max, training_set_min))]).T

    ### categorical data
    encoder = OneHotEncoder(categories='auto')
    cols = config['inputs']['categorical']
    encoder.fit(building_energy[cols].to_numpy())
    categorical = encoder.transform(building_energy[cols].to_numpy()).toarray()

    ### build training, validation and test sets
    # length of history to use for prediction
    history_size = samples_per_day // int(config['history_size'] // 24)
    # how many steps to predict in the future
    target_size = int(config['future_size'] * 60 / time_step)
    # how many steps to look ahead
    steps_ahead = int(config['hours_ahead'] * 60 / time_step)

    X = np.concatenate((data_scaled, categorical), axis=1)
    x = {}
    y = {}
    x['training'],   y['training']   = make_data_blocks(X, 0, train_split, history_size, target_size,
                                                        steps_ahead, target=X[:,0])
    x['validation'], y['validation'] = make_data_blocks(X, train_split, validation_split, history_size,
                                                        target_size, steps_ahead, target=X[:,0])
    x['test'],       y['test']       = make_data_blocks(X, validation_split, None, history_size,
                                                        target_size, steps_ahead, target=X[:,0])

    for key,value in x.items():
        shp = value.shape
        print(f'Shape of the {key} set: ({shp[0]},{shp[1]},{shp[2]})')

    batch_size = config['batch_size']
    buffer_size = config['buffer_size'] if 'buffer_size' in config else 10000
    training_dataset = tf.data.Dataset.from_tensor_slices((x['training'], y['training']))
    training_dataset = training_dataset.cache().shuffle(buffer_size).batch(batch_size).repeat()
    validation_dataset = tf.data.Dataset.from_tensor_slices((x['validation'], y['validation']))
    validation_dataset = validation_dataset.batch(batch_size).repeat()

    try:
        steps_per_epoch = config['steps_per_epoch']
    except:
        N_training_traces, N_samples, N_vars = x['training'].shape
        steps_per_epoch = N_training_traces // batch_size
        print(f'{steps_per_epoch}')

    validation_steps = config['validation_steps'] if 'validation_steps' in config else 50

    if 'learning_rate_schedule' in config and config['learning_rate_schedule']['name'] is not None:
        lr_schedule = config['learning_rate_schedule'][config['learning_rate_schedule']['name']]
        lr_schedule['name'] = config['learning_rate_schedule']['name']
    else:
        lr_schedule = None

    ### store all the parameters
    output_path = args.output_dir + '/LSTM/' + experiment_key
    parameters = config.copy()
    parameters['seed']              = seed
    parameters['training_set_min']  = training_set_min
    parameters['training_set_max']  = training_set_max
    parameters['steps_per_epoch']   = steps_per_epoch
    parameters['validation_steps']  = validation_steps
    parameters['buffer_size']       = buffer_size
    parameters['N_training_traces'] = N_training_traces
    parameters['N_samples']         = N_samples
    parameters['N_vars']            = N_vars
    parameters['output_path']       = output_path

    ### build the network
    optimizer_pars = config['optimizer'][config['optimizer']['name']]
    optimizer_pars['name'] = config['optimizer']['name']
    model, optimizer, loss = build_model(x['training'].shape,
                                         config['model_arch'],
                                         config['loss_function'],
                                         optimizer_pars,
                                         lr_schedule)

    model.summary()

    if log_to_comet:
        experiment.log_parameters(parameters)

    try:
        cb_pars = []
        for name in config['callbacks']['names']:
            cb_pars.append(config['callbacks'][name])
            cb_pars[-1]['name'] = name
    except:
        cb_pars = None

    if log_to_comet:
        # add a bunch of tags to the experiment
        experiment.add_tag('LSTM')
        experiment.add_tag('history={}'.format(config['history_size']))
        experiment.add_tag('future={}'.format(config['future_size']))
        experiment.add_tag('ahead={}'.format(config['hours_ahead']))
        experiment.add_tag('{}_layers'.format(config['model_arch']['N_layers']))
        experiment.add_tag('{}_neurons'.format(config['model_arch']['N_units']))
        try:
            experiment.add_tag(config['learning_rate_schedule']['name'].split('_')[0] + '_lr')
        except:
            pass


    ### train the network
    history = train_model(model, training_dataset, validation_dataset,
                          config['N_epochs'], steps_per_epoch,
                          validation_steps, output_path,
                          experiment, cb_pars, verbose = 1)

    checkpoint_path = output_path + '/checkpoints'
    
    ### find the best model based on the validation loss
    checkpoint_files = glob.glob(checkpoint_path + '/*.h5')
    val_loss = [float(file[:-3].split('-')[-1]) for file in checkpoint_files]
    best_checkpoint = checkpoint_files[np.argmin(val_loss)]
    best_model = models.load_model(best_checkpoint)

    for key in x:
        loss = model.evaluate(x[key], y[key], verbose=0)
        print(f'Loss on the {key} set: {loss:.4f}.')

    best_model.save(output_path)
    pickle.dump(parameters, open(output_path + '/parameters.pkl', 'wb'))
    pickle.dump(history.history, open(output_path + '/history.pkl', 'wb'))

    ### plot a graph of the network topology
    keras.utils.plot_model(model, output_path + '/network_topology.png', show_shapes=True, dpi=300)

    if log_to_comet:
        experiment.log_model('best_model', output_path + '/saved_model.pb')
        experiment.log_image(output_path + '/network_topology.png', 'network_topology')

