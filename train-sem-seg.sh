#!/bin/bash
exp='conditional-lite-sepmetric-class-sbdd-'$1
model='cofeat-late-lite-unshared'
dataset='sbdd'
datatype='late-conditional-class'
split='trainaug'
val_dataset='voc'
val_split='valid'
class_group=$1
lr=1e-6
max_iter=100000
seed=1337
gpu=$2
python -u train.py $exp --model $model --dataset $dataset --datatype $datatype --split $split --val_dataset $val_dataset --val_split $val_split --class_group $class_group --lr $lr --max_iter $max_iter --seed $seed --gpu $gpu --no-eval

