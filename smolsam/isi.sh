#!/bin/bash

LR="0.00005"
PYTHONPATH=. python smolsam/train_one_gpu.py --train-patch-emb --lr $LR -t 5e-5.train-patch-emb
PYTHONPATH=. python smolsam/train_one_gpu.py --from-scratch --lr $LR -t 5e-5.from-scratch
PYTHONPATH=. python smolsam/train_one_gpu.py --train-on-all-data --lr $LR -t 5e-5.train-on-all-data

LR="0.00001"
PYTHONPATH=. python smolsam/train_one_gpu.py --train-patch-emb --lr $LR -t 1e-5.train-patch-emb
PYTHONPATH=. python smolsam/train_one_gpu.py --from-scratch --lr $LR -t 1e-5.from-scratch
PYTHONPATH=. python smolsam/train_one_gpu.py --train-on-all-data --lr $LR -t 1e-5.train-on-all-data
