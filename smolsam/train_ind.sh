#!/bin/bash

#LOG_PATH="/koko/train_out/HisMap/SMOL/gauss_siegfried"
LOG_PATH="/home/yuan/code/HisMap/smolsam/logs"
DATA_PATH="/home/yuan/data/HisMap/SMOLdata"


train () {
  ACCUM_STEP=$1
  RANK=$2
  EP=$3
  DATA=$4
  FG=0.1
  TNAME="$DATA"_e"$EP"_r"$RANK"_a"$ACCUM_STEP"_fg"$FG"
  echo DATASET: $DATA
  echo FG: $FG
  echo RANK: $RANK
  echo EP: $EP
  echo ACCUM_STEP: $ACCUM_STEP
  echo TNAME: $TNAME


  # TRAIN
  PYTHONPATH=. python smolsam/train_one_gpu.py \
  -d $DATA_PATH \
  --data-info all \
  --data-mode train_"$DATA" \
  -t $TNAME --disable-runid --lr 5e-4 --lr-scheduler exp -e $EP -r $RANK -a $ACCUM_STEP -fg $FG

  # TEST
  PYTHONPATH=. python smolsam/inference.py \
  -d $DATA_PATH \
  -w $LOG_PATH/$TNAME/mapsam_model_best.pth \
  -o $LOG_PATH/$TNAME \
  --data-info all \
  --data-mode test_"$DATA" -r $RANK -fg $FG -s


}

train 4 16 200 siegfried_railway
train 4 16 200 siegfried_vineyard
train 4 16 200 hameln
train 4 16 200 bayern
