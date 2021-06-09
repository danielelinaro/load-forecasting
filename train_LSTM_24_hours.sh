#!/bin/bash

for hours in `seq 0 23` ; do
    python3 train_LSTM.py --hours-ahead $hours --max-cores 4 config/LSTM_training_config.json > training_$hours.log
done

