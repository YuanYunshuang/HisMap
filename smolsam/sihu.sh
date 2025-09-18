#!/bin/bash

LOG_PATH="/mars/projects20/HisMap/smolsam/logs"
DATA_PATH="/koko/datasets/SMOL"


train () {
  ACCUM_STEP=$1
  RANK=$2
  EP=$3
  NSHOT=$4
  FG=0.05
  TNAME=gauss_siegfried_"$NSHOT"shot_e"$EP"_r"$RANK"_a"$ACCUM_STEP"_fg"$FG"
  echo FG: $FG
  echo RANK: $RANK
  echo EP: $EP
  echo ACCUM_STEP: $ACCUM_STEP
  echo TNAME: $TNAME


  # TRAIN
  PYTHONPATH=. python smolsam/train_one_gpu.py \
  -d $DATA_PATH \
  --data-info all \
  --data-mode train_"$NSHOT"shot \
  -t $TNAME --disable-runid --lr 5e-4 --lr-scheduler exp -e $EP -r $RANK -a $ACCUM_STEP -fg $FG -itr 100000

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

train 4 16 200 100