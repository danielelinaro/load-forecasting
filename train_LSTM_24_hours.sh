#!/bin/bash

for hours in `seq 0 23` ; do
    python3 train_LSTM.py --max-cores 4 --hours-ahead $hours config/LSTM_training_config.json > training_$hours.log
done

