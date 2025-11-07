#!/bin/bash

PYTHONPATH=. python syntra/train_one_gpu.py --config syntra_chaos.yaml --resume syntra/logs/syntra_chaos/model_latest.pth
PYTHONPATH=. python syntra/train_one_gpu.py --config syntra_facies.yaml