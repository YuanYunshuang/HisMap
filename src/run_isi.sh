#!/bin/bash

for i in {5..1}
do
PYTHONPATH=. python src/run_lgln_2023.py --mode train \
 --config src/config/lgln/seg5cls_focal_2023.yaml \
--load-from /home/yuan/data/out/LGLN2023/2023/epoch20.pth \
 --log-dir /home/yuan/data/out/LGLN2023 --unc-thr 0."$i"

done


