#!/bin/bash

LOG_PATH="/mars/projects20/HisMap/smolsam/logs"
DATA_PATH="/koko/datasets/SMOL"

#  # TRAIN
#PYTHONPATH=. python smolsam/train_one_gpu.py \
#-d $DATA_PATH \
#--data-info all \
#--data-mode train_all \
#-t gauss_siegfried_100k --disable-runid --lr 5e-4 --lr-scheduler exp -e 10 -itr 100000

PYTHONPATH=. python smolsam/inference.py \
-w $LOG_PATH/gauss_siegfried/medsam_model_best.pth \
-o $LOG_PATH/gauss_siegfried \
--data-info all \
--data-mode test

PYTHONPATH=. python smolsam/inference.py \
-w $LOG_PATH/gauss_siegfried/medsam_model_best.pth \
-o $LOG_PATH/gauss_siegfried \
--data-info all \
--data-mode siegfried_railway

PYTHONPATH=. python smolsam/inference.py \
-w $LOG_PATH/gauss_siegfried/medsam_model_best.pth \
-o $LOG_PATH/gauss_siegfried \
--data-info all \
--data-mode siegfried_vineyard