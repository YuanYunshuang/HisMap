#!/bin/bash

#LOG_PATH="/koko/train_out/HisMap/SMOL/gauss_siegfried"
LOG_PATH="/home/yuan/code/HisMap/smolsam/logs"
DATA_PATH="/home/yuan/data/HisMap/SMOLdata"


train () {
  ACCUM_STEP=$1
  RANK=$2
  ITR=$3
  FG=0.02
  TNAME=fewshot_hameln2_r"$RANK"_a"$ACCUM_STEP"_fg"$FG"_itr"$ITR"
  echo FG: $FG
  echo RANK: $RANK
  echo EP: $EP
  echo ACCUM_STEP: $ACCUM_STEP
  echo ITR: ITR
  echo TNAME: $TNAME


#  # TRAIN
#  PYTHONPATH=. python smolsam/train_one_gpu.py \
#  -d $DATA_PATH \
#  --data-info hameln2 \
#  --data-mode train_hameln2 \
#  --resume /home/yuan/code/HisMap/smolsam/logs/gauss_siegfried_e200_r16_a4_fg0.1/mapsam_model_best.pth \
#  -t $TNAME --disable-runid --lr 5e-4 --lr-scheduler exp -r $RANK -a $ACCUM_STEP -fg $FG -itr $ITR -cj 0 -mli 5


  PYTHONPATH=. python smolsam/inference.py \
  -d $DATA_PATH \
  -w $LOG_PATH/$TNAME/mapsam_model_best.pth \
  -o $LOG_PATH/$TNAME \
  --data-info hameln2 \
  --data-mode test_hameln2 -r $RANK -fg 0.01 -s


}

train 4 16 50000