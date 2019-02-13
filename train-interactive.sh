#!/bin/bash
exp='interactive-inst-sbdd'
model='dios-late-glob'
dataset='sbdd'
datatype='interactive'
split='train'
val_dataset='sbdd'
val_split='val'
lr=1e-5
max_iter=100000
seed=1337
gpu=$1
python -u train.py $exp --model $model --dataset $dataset --datatype $datatype --split $split --val_dataset $val_dataset --val_split $val_split --lr $lr --max_iter $max_iter --seed $seed --gpu $gpu --no-eval
