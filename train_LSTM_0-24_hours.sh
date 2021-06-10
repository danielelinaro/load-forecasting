#!/bin/bash

future=0.5
ncores=2
nparallel=8
hours=0
while (( $(echo "$hours < 24" | bc -l) )) ; do
    for i in `seq $nparallel` ; do
	python3 train_LSTM.py --max-cores $ncores --hours-ahead $hours config/LSTM_training_config.json > training_$hours.log
	hours=$(echo "$hours+$future" | bc)
    done
    wait
done
