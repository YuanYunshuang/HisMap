#!/bin/bash

for i in {6..9}
do
 PYTHONPATH=. python src/run_lgln_2023.py --mode train \
 --config src/config/lgln/seg5cls_focal_2023.yaml \
 --data-path /media/yuan/luna1/data/HisMAP/LGLN/dataset \
 --load-from /media/yuan/luna1/lgln-2023/2023/epoch20.pth \
 --log-dir /media/yuan/luna1/lgln-2023 --unc-thr 0."$i"
done

