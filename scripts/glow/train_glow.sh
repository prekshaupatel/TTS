#!/bin/bash

# $1 -> 'male'

gender=$1

cp ~/vakyansh-tts/checkpoints/glow/$gender/config.json ../../config/glow/$gender.json

config='../../config/glow/'$gender'.json'
modeldir='../../checkpoints/glow/'$gender
logdir='../../logs/glow/'$gender
init=0  # 1 if start from scratch. 0 if start from last checkpoint


####################################################

if [[ $init -eq 1 ]]
then
  python ../../src/glow_tts/init.py -c $config -m $modeldir -l $logdir
fi
python ../../src/glow_tts/train.py -c $config -m $modeldir -l $logdir
