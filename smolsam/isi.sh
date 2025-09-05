#!/bin/bash

LOG_PATH="/home/yuan/code/HisMap/smolsam/logs"
DATA_PATH="/home/yuan/data/HisMap/SMOLdata"


d="vineyard_fewshot10"

for x in "30000" "40000"; do
  # TRAIN
  PYTHONPATH=. python smolsam/train_one_gpu.py \
  --resume $LOG_PATH/siegfried_railway/medsam_model_latest.pth \
  -d $DATA_PATH\
  --data-info siegfried_train \
  --data-mode train_siegfried_$d \
  -t siegfried_railway_$d"_itr"$x --disable-runid --lr 5e-3 --lr-scheduler exp -e 10 -itr $x
  # TEST
  PYTHONPATH=. python smolsam/inference.py \
  -w $LOG_PATH/siegfried_railway_$d"_itr"$x/medsam_model_best.pth \
  -o $LOG_PATH/siegfried_railway_$d"_itr"$x \
  --data-info siegfried_test \
  --data-mode test_siegfried_vineyard
done