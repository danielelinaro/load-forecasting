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
import pandas as pd
import matplotlib.pyplot as plt

from comet_ml import Experiment
from comet_ml.api import API
from comet_ml.query import Tag

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, callbacks, models
import tensorflow_addons as tfa
from sklearn.preprocessing import OneHotEncoder

import colorama as cm
print_error   = lambda msg: print(f'{cm.Fore.RED}'    + msg + f'{cm.Style.RESET_ALL}')
print_warning = lambda msg: print(f'{cm.Fore.YELLOW}' + msg + f'{cm.Style.RESET_ALL}')
print_msg     = lambda msg: print(f'{cm.Fore.GREEN}'  + msg + f'{cm.Style.RESET_ALL}')

comet_workspace = 'danielelinaro'
comet_project_name = 'load-forecasting'

LEARNING_RATE = []

__all__ = ['make_data_blocks', 'extract_time_step', 'add_minute_and_workday',
           'average_data', 'compute_stats', 'make_dataset']

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


def extract_time_step(df):
    t0 = datetime.datetime.combine(datetime.date.today(),
                                   df['datetime'][0].to_pydatetime().time())
    t1 = datetime.datetime.combine(datetime.date.today(),
                                   df['datetime'][1].to_pydatetime().time())
    return int((t1 - t0).total_seconds() / 60) # [min]


def add_minute_and_workday(df):
    df = df.copy()
    df['minute'] = [(dt.hour * 60 + dt.minute) for dt in df['datetime']]
    df['workday'] = np.logical_not(df['weekend'] | df['holiday'])
    cols = df.columns.tolist()
    cols = cols[:1] + cols[-2:] + cols[1:-2]
    return df[cols]


def average_data(df, time_step, orig_time_step, columns):
    df = df.copy()
    avg_step = time_step // orig_time_step
    for column in columns:
        df[column + '_averaged'] = df[column].rolling(window=avg_step).mean()
    df = df[avg_step - 1 : : avg_step]
    return df


def compute_stats(df, time_step):
    samples_per_day = 24 * 60 // time_step
    n_samples = df.shape[0]
    n_days = n_samples // samples_per_day
    return n_days, samples_per_day


def make_dataset(df, cols_continuous,  cols_categorical,
                 training_set_max, training_set_min,
                 n_days, samples_per_day, encoder=None, full_output=False):
    x = df[cols_continuous].to_numpy(dtype=np.float32)
    x_scaled = np.array([-1 + 2 * (x[:,i] - m) / (M - m)
                         for i,(M,m) in enumerate(zip(training_set_max, training_set_min))]).T
    if encoder is None:
        encoder = OneHotEncoder(categories='auto')
        encoder.fit(df[cols_categorical].to_numpy())
    categorical = encoder.transform(df[cols_categorical].to_numpy()).toarray()
    if full_output:
        return np.concatenate((x_scaled, categorical), axis=1), encoder
    return np.concatenate((x_scaled, categorical), axis=1)


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
    parser.add_argument('--max-cores',  default=None,  type=int, help='maximum number of cores to be used by Keras)')
    parser.add_argument('-o', '--output-dir',  default='experiments',  type=str, help='output directory')
    parser.add_argument('--new-data', action='store_true', help='continue training with new data')
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

    if args.max_cores is not None:
        if args.max_cores > 0:
            tf.config.threading.set_inter_op_parallelism_threads(args.max_cores)
            tf.config.threading.set_intra_op_parallelism_threads(args.max_cores)
            print_msg(f'Maximum number of cores set to {args.max_cores}.')
        else:
            print_warning('Maximum number of cores must be positive.')

    if args.hours_ahead is not None:
        config['hours_ahead'] = args.hours_ahead
        print_msg('Hours ahead for prediction: {:.1f}.'.format(config['hours_ahead']))

    log_to_comet = not args.no_comet

    if log_to_comet:
        ### create a CometML experiment
        experiment = Experiment(
            api_key = os.environ['COMET_API_KEY'],
            project_name = comet_project_name,
            workspace = comet_workspace
        )
        experiment_key = experiment.get_key()
    else:
        experiment = None
        experiment_key = strftime('%Y%m%d-%H%M%S', localtime())

    time_step = config['time_step']

    if 'building_temperature' in config['inputs']['continuous']:
        with_building_temperature = True
    else:
        with_building_temperature = False

    if len(config['inputs']['continuous']) == 1 and config['inputs']['continuous'][0] == 'building_consumption':
        consumption_only = True
    else:
        consumption_only = False

    if args.new_data:
        ### here we load the weights from a previous experiment
        api = API(api_key = os.environ['COMET_API_KEY'])
        workspace = comet_workspace
        project_name = comet_project_name

        # the tags used to find the right experiment
        query = Tag('LSTM') & \
            Tag('{}_layers'.format(config['model_arch']['N_layers'])) & \
            Tag('{}_neurons'.format(config['model_arch']['N_units'])) & \
            Tag('ahead={:.1f}'.format(config['hours_ahead'])) & \
            Tag('future={:.2f}'.format(config['future_size'])) & \
            Tag('history={}'.format(config['history_size']))
        if with_building_temperature:
            query &= Tag('building_temperature')

        # find all the experiment that match the set of tags
        completed_experiments = api.query(workspace, project_name, query, archived=False)
        if not with_building_temperature:
            completed_experiments = [expt for expt in completed_experiments if all([tag != 'building_temperature' \
                                                                                    for tag in expt.get_tags()])]

        # iterate over all experiments to find the one with the smallest validation loss
        print(f'{len(completed_experiments)} experiments match the query tags.')
        min_val_loss = 100
        for completed_experiment in completed_experiments:
            metrics = completed_experiment.get_metrics()
            loss = np.array([float(m['metricValue']) for m in metrics if m['metricName'] == 'val_loss'])
            if loss.min() < min_val_loss:
                val_loss = loss
                min_val_loss = loss.min()
                tags = completed_experiment.get_tags()
                best_experiment_ID = completed_experiment.id

        # load the best model
        experiments_path = 'experiments/LSTM/'
        checkpoint_path = experiments_path + best_experiment_ID + '/checkpoints/'
        checkpoint_files = glob.glob(checkpoint_path + '*.h5')
        epochs = [int(os.path.split(file)[-1].split('.')[1].split('-')[0])
                  for file in checkpoint_files]
        best_checkpoint = checkpoint_files[epochs.index(np.argmin(val_loss) + 1)]
        trained_model = keras.models.load_model(best_checkpoint, compile=False)
        trained_model_pars = pickle.load(open(experiments_path + best_experiment_ID + '/parameters.pkl', 'rb'))
        print_msg(f'Loaded best checkpoint from experiment {best_experiment_ID[:9]} ({len(val_loss):3d} epochs) ' + \
                  f'validation loss: {min_val_loss:.4f}\n  tags: {tags}')

    data_files = config['data_files']
    for data_file in data_files:
        if not os.path.isfile(data_file):
            print_error('{}: {}: no such file.'.format(progname, data_file))
            sys.exit(1)

    ### load the data
    full_data = {}
    for data_file in data_files:
        blob = pickle.load(open(data_file, 'rb'))
        for key1 in blob:
            if key1 not in full_data:
                full_data[key1] = blob[key1]
            else:
                for key2,value2 in blob[key1].items():
                    if isinstance(value2, dict):
                        for key3,value3 in blob[key1][key2].items():
                            full_data[key1][key2][key3] = full_data[key1][key2][key3].append(value3, ignore_index=True)
                    elif isinstance(value2, pd.DataFrame):
                        full_data[key1][key2] = full_data[key1][key2].append(value2, ignore_index=True)
                    else:
                        raise Exception(f'Do not know how to deal with object of type {type(value2)}')
    building_energy = full_data['full']['building_energy']
    building_sensor = full_data['full']['building_sensor']
    weather = full_data['full']['weather_data']
    # the dataframe with consumption and generation is the starting point for building
    # the dataset used for training, validation and testing
    data = building_energy.copy()
    data.rename({'consumption': 'building_consumption', 'generation': 'building_generation'}, axis='columns', inplace=True)
    data['building_temperature'] = building_sensor['temperature'].copy()

    orig_time_step = extract_time_step(data)
    data = add_minute_and_workday(data)

    cols_continuous = config['inputs']['continuous']
    use_zones = any(['zone' in col for col in cols_continuous])
    use_sensors = any(['sensor' in col for col in cols_continuous])
    use_weather = any(['weather' in col for col in cols_continuous])

    def merge_data(blob, data, prefix):
        for index,df in blob.items():
            for in_col in df.columns:
                if in_col not in data.columns:
                    out_col = f'{prefix}{index}_{in_col.lower()}'
                    data[out_col] = df[in_col]

    if use_zones:
        merge_data(full_data['full']['zones'], data, 'zone')

    if use_sensors:
        merge_data(full_data['full']['sensors'], data, 'sensor')

    data = average_data(data, time_step, orig_time_step, [col for col in cols_continuous if 'weather' not in col])

    if use_weather:
        df = weather[['datetime','temperature','humidity','radiation']].copy()
        df.rename({col: 'weather_' + col for col in df.columns if col != 'datetime'}, axis='columns', inplace=True)
        data = pd.merge(data, df, how='inner', on=['datetime'])
        for key in 'temperature','humidity','radiation':
            idx, = np.where(np.isnan(data['weather_' + key]))
            gap = np.diff(np.where(np.diff(idx) > 1)).max() * time_step / 60
            print_warning(f'Longest gap in {key} data: {gap:g} hours')

    data.fillna(method='pad', inplace=True)

    n_days, samples_per_day = compute_stats(data, time_step)
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
    print('The full training set contains {} measurements (corresponding to {} days).'.\
          format(train_split, n_days_training))
    print('The validation set contains {} measurements (corresponding to {} days).'.\
          format(validation_split - train_split, n_days_validation))
    print('The test set contains {} measurements (corresponding to {} days).'.\
          format(n_days_test * samples_per_day, n_days_test))

    ### normalize the data
    if config['average_continuous_inputs']:
        cols_continuous = [col + '_averaged' if 'weather' not in col else col for col in cols_continuous]
    cols_categorical = config['inputs']['categorical']

    cols_idx = [data.columns.get_loc(col) for col in cols_continuous]
    training_set_max = data.iloc[:train_split, cols_idx].max().to_numpy(dtype=np.float32)
    training_set_min = data.iloc[:train_split, cols_idx].min().to_numpy(dtype=np.float32)

    X = make_dataset(data, cols_continuous, cols_categorical, training_set_max, training_set_min, n_days, samples_per_day)

    ### build training, validation and test sets
    # length of history to use for prediction
    history_size = samples_per_day * int(config['history_size'] // 24)
    # how many steps to predict in the future
    target_size = int(config['future_size'] * 60 / time_step)
    # how many steps to look ahead
    steps_ahead = int(config['hours_ahead'] * 60 / time_step)

    x = {}
    y = {}
    x['training'],   y['training']   = make_data_blocks(X, 0, train_split, history_size, target_size,
                                                        steps_ahead, target=X[:,0])
    x['validation'], y['validation'] = make_data_blocks(X, train_split, validation_split, history_size,
                                                        target_size, steps_ahead, target=X[:,0])
    x['test'],       y['test']       = make_data_blocks(X, validation_split, None, history_size,
                                                        target_size, steps_ahead, target=X[:,0])

    if args.new_data:
        # use as training set only the data that the network has not seen previously
        x['training'] = x['training'][trained_model_pars['N_training_traces']:, :, :]
        y['training'] = y['training'][trained_model_pars['N_training_traces']:]

    for key,value in x.items():
        shp = value.shape
        print(f'Shape of the {key} set: ({shp[0]},{shp[1]},{shp[2]})')

    batch_size = config['batch_size']
    buffer_size = config['buffer_size'] if 'buffer_size' in config else 10000
    training_dataset = tf.data.Dataset.from_tensor_slices((x['training'], y['training']))
    training_dataset = training_dataset.cache().shuffle(buffer_size).batch(batch_size).repeat()
    validation_dataset = tf.data.Dataset.from_tensor_slices((x['validation'], y['validation']))
    validation_dataset = validation_dataset.batch(batch_size).repeat()

    N_training_traces, N_samples, N_vars = x['training'].shape
    if N_training_traces == 0:
        print_error('No traces in training set.')
        sys.exit(1)

    try:
        steps_per_epoch = config['steps_per_epoch']
    except:
        steps_per_epoch = N_training_traces // batch_size

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
    if args.new_data:
        parameters['N_training_traces'] += trained_model_pars['N_training_traces']

    ### build the network
    optimizer_pars = config['optimizer'][config['optimizer']['name']]
    optimizer_pars['name'] = config['optimizer']['name']
    model, optimizer, loss = build_model(x['training'].shape,
                                         config['model_arch'],
                                         config['loss_function'],
                                         optimizer_pars,
                                         lr_schedule)
    if args.new_data:
        model.set_weights(trained_model.get_weights())
        print_msg('Set model weights from previous training.')

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
        experiment.add_tag('future={:.2f}'.format(config['future_size']))
        experiment.add_tag('ahead={:.1f}'.format(config['hours_ahead']))
        experiment.add_tag('{}_layers'.format(config['model_arch']['N_layers']))
        experiment.add_tag('{}_neurons'.format(config['model_arch']['N_units']))
        experiment.add_tag('_'.join([os.path.splitext(os.path.basename(data_file))[0] \
                                     for data_file in config['data_files']]))
        if with_building_temperature:
            experiment.add_tag('building_temperature')
        if consumption_only:
            experiment.add_tag('consumption_only')
        if args.new_data:
            experiment.add_tag('initialized_weights')
        else:
            experiment.add_tag('random_initial_weights')
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

