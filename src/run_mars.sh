#!/bin/bash

for i in {4..1}
do
 PYTHONPATH=. python src/run_lgln.py --mode train \
 --config src/config/lgln/seg5cls_focal.yaml \
 --data-path /media/yuan/luna1/data/HisMAP/LGLN/dataset \
 --load-from /media/yuan/luna1/lgln/all/epoch20.pth \
 --log-dir /media/yuan/luna1/lgln-2023 --unc-thr 0."$i"
done

