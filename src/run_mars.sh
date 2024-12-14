#!/bin/bash

for i in {5..9}
do
 PYTHONPATH=. python src/run_lgln.py --mode train \
 --config src/config/lgln/seg5cls_focal_bi.yaml \
 --data-path /media/yuan/luna1/data/HisMAP/LGLN/dataset \
 --load-from /media/yuan/luna1/lgln/all/epoch20.pth \
 --log-dir /media/yuan/luna1/lgln --unc-thr 0."$i"
done

