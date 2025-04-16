#!/bin/bash

LR="0.00005"
PYTHONPATH=. python smolsam/train_one_gpu.py --use-amp --lr $LR -t 5e-5.use-amp
PYTHONPATH=. python smolsam/train_one_gpu.py --use-amp --train-patch-emb --lr $LR -t 5e-5.amp.train-patch-emb
PYTHONPATH=. python smolsam/train_one_gpu.py --use-amp --from-scratch --lr $LR -t 5e-5.amp.from-scratch
PYTHONPATH=. python smolsam/train_one_gpu.py --use-amp --train-on-all-data --lr $LR -t 5e-5.amp.train-on-all-data

LR="0.00001"
PYTHONPATH=. python smolsam/train_one_gpu.py --use-amp --use-amp --lr $LR -t 1e-5.use-amp
PYTHONPATH=. python smolsam/train_one_gpu.py --use-amp --from-scratch --lr $LR -t 1e-5.amp.from-scratch
PYTHONPATH=. python smolsam/train_one_gpu.py --use-amp --train-on-all-data --lr $LR -t 1e-5.amp.train-on-all-data
