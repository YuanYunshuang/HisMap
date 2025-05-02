#!/bin/bash

PYTHONPATH=. python smolsam/train_one_gpu.py -d /media/yuan/luna1/data/SMOLdata --lr 0.00005 -t 5e-5
PYTHONPATH=. python smolsam/train_one_gpu.py -d /media/yuan/luna1/data/SMOLdata --lr 0.00001 -t 1e-5
PYTHONPATH=. python smolsam/train_one_gpu.py -d /media/yuan/luna1/data/SMOLdata --lr 0.0001 -t 1e-4

