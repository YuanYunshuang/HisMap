#!/bin/bash

#LOG_PATH="/koko/train_out/HisMap/SMOL/gauss_siegfried"
LOG_PATH="/home/yuan/code/HisMap/smolsam/logs"
DATA_PATH="/home/yuan/data/HisMap/SMOLdata"


train () {
  ACCUM_STEP=$1
  RANK=$2
  EP=$3
  FG=0.1
  TNAME=gauss_siegfried_e"$EP"_r"$RANK"_a"$ACCUM_STEP"_fg"$FG"
  echo FG: $FG
  echo RANK: $RANK
  echo EP: $EP
  echo ACCUM_STEP: $ACCUM_STEP
  echo TNAME: $TNAME


  # TRAIN
  PYTHONPATH=. python smolsam/train_one_gpu.py \
  -d $DATA_PATH \
  --data-info all \
  --data-mode train_all \
  -t $TNAME --disable-runid --lr 5e-4 --lr-scheduler exp -e $EP -r $RANK -a $ACCUM_STEP -fg $FG

  # TEST
  for d in "hameln" "bayern" "siegfried_railway" "siegfried_vineyard"; do
  PYTHONPATH=. python smolsam/inference.py \
  -d $DATA_PATH \
  -w $LOG_PATH/$TNAME/mapsam_model_best.pth \
  -o $LOG_PATH/$TNAME \
  --data-info all \
  --data-mode test_$d -r $RANK -fg $FG -s
  done

}

train 4 32 200
train 8 16 200