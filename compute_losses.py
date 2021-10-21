
import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from train_LSTM import *

if __name__ == '__main__':

    data_files = ['data/data1.pkl']
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
    # the dataframe with consumption and generation is the starting point for building
    # the dataset used for training, validation and testing
    data = building_energy.copy()
    data.rename({'consumption': 'building_consumption', 'generation': 'building_generation'}, axis='columns', inplace=True)
    data['building_temperature'] = building_sensor['temperature'].copy()

    time_step = 15
    orig_time_step = extract_time_step(data)
    data = add_minute_and_workday(data)
    data = average_data(data, time_step, orig_time_step,
                        ['building_consumption', 'building_generation', 'building_temperature'])
    data.fillna(method='pad', inplace=True)

    n_days, samples_per_day = compute_stats(data, time_step)
    print(f'Original time step: {orig_time_step} minutes.')
    print(f'Time step: {time_step} minutes.')
    print(f'Number of days: {n_days}.')
    print(f'Samples per day: {samples_per_day}.')

    expt_folder = 'experiments/LSTM'
    folders = glob.glob(expt_folder + '/*')

    RE_EVALUATE = False

    for folder in folders:

        try:
            if os.path.isfile(folder + '/losses.pkl') and not RE_EVALUATE:
                continue

            pars = pickle.load(open(folder + '/parameters.pkl','rb'))

            if ('data_file' in pars and [pars['data_file']] != data_files) or \
               ('data_files' in pars and pars['data_files'] != data_files):
                # different data set
                continue

            print(os.path.split(folder)[-1])

            ### how many days to use for training, test and validation
            n_days_training = int(pars['data_split']['training'] * n_days)
            n_days_test = int(pars['data_split']['test'] * n_days)
            n_days_validation = n_days - n_days_training - n_days_test
            train_split = n_days_training * samples_per_day
            validation_split = (n_days_training + n_days_validation) * samples_per_day

            ### normalize the data
            if pars['average_continuous_inputs']:
                continuous_inputs = [inp + '_averaged' for inp in pars['inputs']['continuous']]
            else:
                continuous_inputs = pars['continuous_inputs']

            prefix = 'building_'
            continuous_inputs = [inp if prefix in inp else prefix + inp for inp in continuous_inputs]

            X = make_dataset(data,
                             continuous_inputs,
                             pars['inputs']['categorical'],
                             pars['training_set_max'],
                             pars['training_set_min'],
                             n_days,
                             samples_per_day)

            # length of history to use for prediction
            history_size = samples_per_day * int(pars['history_size'] // 24)
            # how many steps to predict in the future
            target_size = int(pars['future_size'] * 60 / time_step)
            # how many steps to look ahead
            steps_ahead = int(pars['hours_ahead'] * 60 / time_step)

            x = {}
            y = {}
            x['validation'], y['validation'] = make_data_blocks(X, train_split, validation_split, history_size,
                                                                target_size, steps_ahead, target=X[:,0])
            x['test'],       y['test']       = make_data_blocks(X, validation_split, None, history_size,
                                                                target_size, steps_ahead, target=X[:,0])

            ### find the best model based on the validation loss
            checkpoint_path = folder + '/checkpoints'
            checkpoint_files = glob.glob(checkpoint_path + '/*.h5')
            #val_loss_computed = [tf.keras.models.load_model(cp_file).evaluate(x['validation'],
            #                                                                  y['validation'],
            #                                                                  batch_size=pars['batch_size'])
            #                     for cp_file in checkpoint_files]
            val_loss = [float(file[:-3].split('-')[-1]) for file in checkpoint_files]
            best_checkpoint = checkpoint_files[np.argmin(val_loss)]
            best_model = tf.keras.models.load_model(best_checkpoint)

            losses = {}
            for key in x:
                losses[key] = best_model.evaluate(x[key], y[key], batch_size=pars['batch_size'])
                print(f'Loss on the {key} set: {losses[key]:.4f}.')

            pickle.dump(losses, open(folder + '/losses.pkl', 'wb'))

        except:
            print('uh oh...')

