#!/bin/bash
exp='cofeat-late-lite-sbdd-class-sepmetric-'$1
model='cofeat-late-lite-unshared'
data='voc'
datatype='late-conditional-class'
split='valid'
count=1
shot=1
seed=1337
gpu=$2
python -u evaluate.py $exp --model $model --dataset $data --datatype $datatype --split $split --count $count --shot $shot --seed $seed --gpu $gpu --save_seg
