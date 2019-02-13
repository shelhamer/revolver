#!/bin/bash
exp='interactive-inst-sbdd'
model='dios-late-glob'
data='voc'
datatype='interactive'
split='valid'
count=1
shot=1
seed=1337
gpu=$1
python -u evaluate.py $exp --model $model --weights $weights --dataset $data --datatype $datatype --split $split --shot $shot --count $count --seed $seed --gpu $gpu --save_seg


