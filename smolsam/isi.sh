#!/bin/bash

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


##for d in "railway_fewshot10"; do
#d="railway_fewshot10"
#  # TRAIN
#  PYTHONPATH=. python smolsam/train_one_gpu.py \
#  -d /home/yuan/data/HisMap/SMOLdata \
#  --data-info siegfried_train \
#  --data-mode train_siegfried_$d \
#  -t siegfried_$d  --disable-runid
#  # TEST
#  PYTHONPATH=. python smolsam/inference.py \
#  -w /home/yuan/code/HisMap/smolsam/logs/siegfried_$d/medsam_model_best.pth \
#  -o /home/yuan/code/HisMap/smolsam/logs/siegfried_$d/results \
#  --data-info siegfried_test \
#  --data-mode test_siegfried_railway
##done
#
#
##for d in "vineyard_fewshot10"; do
#d="vineyard_fewshot10"
#  # TRAIN
#  PYTHONPATH=. python smolsam/train_one_gpu.py \
#  -d /home/yuan/data/HisMap/SMOLdata \
#  --data-info siegfried_train \
#  --data-mode train_siegfried_$d \
#  -t siegfried_$d --disable-runid
#  # TEST
#  PYTHONPATH=. python smolsam/inference.py \
#  -w /home/yuan/code/HisMap/smolsam/logs/siegfried_$d/medsam_model_best.pth \
#  -o /home/yuan/code/HisMap/smolsam/logs/siegfried_$d/results \
#  --data-info siegfried_test \
#  --data-mode test_siegfried_vineyard
##done

d="vineyard_fewshot10"
  # TRAIN
  PYTHONPATH=. python smolsam/train_one_gpu.py \
  --resume /home/yuan/code/HisMap/smolsam/logs/siegfried_railway/medsam_model_latest.pth \
  -d /home/yuan/data/HisMap/SMOLdata \
  --data-info siegfried_train \
  --data-mode train_siegfried_$d \
  -t siegfried_$d --disable-runid --lr 5e-3 --lr-scheduler exp
  # TEST
  PYTHONPATH=. python smolsam/inference.py \
  -w /home/yuan/code/HisMap/smolsam/logs/siegfried_$d/medsam_model_best.pth \
  -o /home/yuan/code/HisMap/smolsam/logs/siegfried_$d/results \
  --data-info siegfried_test \
  --data-mode test_siegfried_vineyard
