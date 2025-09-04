#!/bin/bash

LOG_PATH="/mars/projects20/HisMap/smolsam/logs"
DATA_PATH="/koko/datasets/smolseg"
## TRAIN
#PYTHONPATH=. python smolsam/train_one_gpu.py \
#-d /home/yuan/data/HisMap/SMOLdata \
#--data-info siegfried_train \
#--data-mode train_siegfried \
#-t siegfried

## TEST
#for d in "railway" "vineyard"; do
#  PYTHONPATH=. python smolsam/inference.py \
#  -w /home/yuan/code/HisMap/smolsam/logs/siegfried/medsam_model_best.pth \
#  -o /home/yuan/code/HisMap/smolsam/logs/siegfried/results \
#  --data-info siegfried_test \
#  --data-mode test_siegfried_$d
#done


for d in "railway" "railway_partial_10" "railway_partial_1" "railway_fewshot10"; do
  # TRAIN
  PYTHONPATH=. python smolsam/train_one_gpu.py \
  -d $DATA_PATH \
  --data-info siegfried_train \
  --data-mode train_siegfried_$d \
  -t siegfried_$d  --disable-runid --lr 5e-3 --lr-scheduler exp
  # TEST
  PYTHONPATH=. python smolsam/inference.py \
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
  -t siegfried_$d --disable-runid --lr 5e-3 --lr-scheduler exp
  # TEST
  PYTHONPATH=. python smolsam/inference.py \
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
-t siegfried_railway_$d --disable-runid --lr 5e-3 --lr-scheduler exp
# TEST
PYTHONPATH=. python smolsam/inference.py \
-w $LOG_PATH/siegfried_$d/medsam_model_best.pth \
-o $LOG_PATH/siegfried_$d/results \
--data-info siegfried_test \
--data-mode test_siegfried_vineyard
