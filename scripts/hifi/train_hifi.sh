#!/bin/bash

# $1 -> 'female'
# $2 -> 60000 (training_epochs)
gender=$1

config='../../config/hifi/config_v1.json'
modeldir='../../checkpoints/hifi/'$gender
logdir='../../logs/hifi/'$gender


####################################################



python ../../src/hifi_gan/train.py \
    --config $config \
    --input_training_file '../../data/hifi/'$gender'/train.txt' \
    --input_validation_file '../../data/hifi/'$gender'/valid.txt' \
    --checkpoint_path $modeldir \
    --logs_path $logdir \
    --checkpoint_interval 1000 \
    --training_epochs $2 \
    --stdout_interval 50
