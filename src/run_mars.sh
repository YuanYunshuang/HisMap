

 PYTHONPATH=. python src/run_lgln.py --mode train \
 --config src/config/lgln/seg5cls_focal.yaml \
 --data-path /media/yuan/luna1/data/HisMAP/LGLN/dataset \
 --load-from /media/yuan/luna1/lgln/epoch20.pth \
 --log-dir /media/yuan/luna1/lgln --unc-thr 0.8

  PYTHONPATH=. python src/run_lgln.py --mode train \
 --config src/config/lgln/seg5cls_focal.yaml \
 --data-path /media/yuan/luna1/data/HisMAP/LGLN/dataset \
 --load-from /media/yuan/luna1/lgln/epoch20.pth \
 --log-dir /media/yuan/luna1/lgln --unc-thr 0.9

