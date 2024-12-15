#!/bin/bash


 PYTHONPATH=. python src/run_lgln_2023.py --mode train \
 --config src/config/lgln/seg5cls_focal_bi.yaml \
 --data-path /media/yuan/luna1/data/HisMAP/LGLN/dataset \
 --load-from /media/yuan/luna1/lgln/all/epoch20.pth \
 --log-dir /media/yuan/luna1/lgln --unc-thr 0.5 --reparse-pseudo-label

for i in {6..9}
do
 PYTHONPATH=. python src/run_lgln_2023.py --mode train \
 --config src/config/lgln/seg5cls_focal_bi.yaml \
 --data-path /media/yuan/luna1/data/HisMAP/LGLN/dataset \
 --load-from /media/yuan/luna1/lgln/all/epoch20.pth \
 --log-dir /media/yuan/luna1/lgln --unc-thr 0."$i" --reparse-pseudo-label
done

