#!/bin/bash

LOG_PATH="/mars/projects20/HisMap/smolsam/logs"
DATA_PATH="/koko/datasets/SMOL"
RANK=32
EP=200
TNAME=gauss_siegfried_e"$EP"_r$RANK

# TRAIN
PYTHONPATH=. python smolsam/train_one_gpu.py \
-d $DATA_PATH \
--data-info all \
--data-mode train_all \
-t "$TNAME" --disable-runid --lr 5e-4 --lr-scheduler exp -e $EP -r $RANK

PYTHONPATH=. python smolsam/inference.py \
-d $DATA_PATH \
-w $LOG_PATH/"$TNAME"/medsam_model_best.pth \
-o $LOG_PATH/"$TNAME" \
--data-info all \
--data-mode test_hameln -r $RANK

PYTHONPATH=. python smolsam/inference.py \
-d $DATA_PATH \
-w $LOG_PATH/"$TNAME"/medsam_model_best.pth \
-o $LOG_PATH/"$TNAME" \
--data-info all \
--data-mode test_bayern -r $RANK

PYTHONPATH=. python smolsam/inference.py \
-d $DATA_PATH \
-w $LOG_PATH/"$TNAME"/medsam_model_best.pth \
-o $LOG_PATH/"$TNAME" \
--data-info all \
--data-mode test_siegfried_railway -r $RANK

PYTHONPATH=. python smolsam/inference.py \
-d $DATA_PATH \
-w $LOG_PATH/"$TNAME"/medsam_model_best.pth \
-o $LOG_PATH/"$TNAME" \
--data-info all \
--data-mode test_siegfried_vineyard -r $RANK
