#!/bin/bash

LOG_PATH="/mars/projects20/HisMap/smolsam/logs"
DATA_PATH="/koko/datasets/Siegfried_Railway_Vineyard/smolseg"

for d in "railway" "railway_partial_10" "railway_partial_1" "railway_fewshot10"; do
  # TRAIN
  PYTHONPATH=. python smolsam/train_one_gpu.py \
  -d $DATA_PATH \
  --data-info siegfried_train \
  --data-mode train_siegfried_$d \
  -t siegfried_$d  --disable-runid --lr 5e-4 --lr-scheduler exp -e 10
  # TEST
  PYTHONPATH=. python smolsam/inference.py \
  -d $DATA_PATH \
  -w $LOG_PATH/siegfried_$d/medsam_model_best.pth \
  -o $LOG_PATH/siegfried_$d/results \
  --data-info siegfried_test \
  --data-mode test_siegfried_railway
done


for d in "vineyard" "vineyard_fewshot10"; do
  # TRAIN
  PYTHONPATH=. python smolsam/train_one_gpu.py \
  -d $DATA_PATH \
  --data-info siegfried_train \
  --data-mode train_siegfried_$d \
  -t siegfried_$d --disable-runid --lr 5e-4 --lr-scheduler exp -e 10
  # TEST
  PYTHONPATH=. python smolsam/inference.py \
  -d $DATA_PATH \
  -w $LOG_PATH/siegfried_$d/medsam_model_best.pth \
  -o $LOG_PATH/siegfried_$d/results \
  --data-info siegfried_test \
  --data-mode test_siegfried_vineyard
done

d="vineyard_fewshot10"
  # TRAIN
PYTHONPATH=. python smolsam/train_one_gpu.py \
--resume $LOG_PATH/siegfried_railway/medsam_model_latest.pth \
-d $DATA_PATH \
--data-info siegfried_train \
--data-mode train_siegfried_$d \
-t siegfried_railway_$d --disable-runid --lr 5e-4 --lr-scheduler exp -e 10
# TEST
PYTHONPATH=. python smolsam/inference.py \
-d $DATA_PATH \
-w $LOG_PATH/siegfried_$d/medsam_model_best.pth \
-o $LOG_PATH/siegfried_$d/results \
--data-info siegfried_test \
--data-mode test_siegfried_vineyard
