#!/bin/bash

LOG_PATH="/mars/projects20/HisMap/smolsam/logs"
DATA_PATH="/koko/datasets/SMOL"
RANK=32

# TRAIN
PYTHONPATH=. python smolsam/train_one_gpu.py \
-d $DATA_PATH \
--data-info all \
--data-mode train_all \
-t gauss_siegfried_200e --disable-runid --lr 5e-4 --lr-scheduler exp -e 200 -r $RANK

PYTHONPATH=. python smolsam/inference.py \
-d $DATA_PATH \
-w $LOG_PATH/gauss_siegfried_200e/medsam_model_best.pth \
-o $LOG_PATH/gauss_siegfried_200e \
--data-info all \
--data-mode test -r $RANK

PYTHONPATH=. python smolsam/inference.py \
-d $DATA_PATH \
-w $LOG_PATH/gauss_siegfried_200e/medsam_model_best.pth \
-o $LOG_PATH/gauss_siegfried_200e \
--data-info all \
--data-mode test_siegfried_railway -r $RANK

PYTHONPATH=. python smolsam/inference.py \
-d $DATA_PATH \
-w $LOG_PATH/gauss_siegfried_200e/medsam_model_best.pth \
-o $LOG_PATH/gauss_siegfried_200e \
--data-info all \
--data-mode test_siegfried_vineyard -r $RANK
