PYTHONPATH=. python src/run_lgln.py --mode train \
 --config src/config/lgln/seg5cls_focal.yaml \
 --load-from /home/yuan/data/out/LGLN/lgln-1973-1975/epoch20.pth  \
 --log-dir /home/yuan/data/out/LGLN --unc-thr 0.3

 PYTHONPATH=. python src/run_lgln.py --mode train \
 --config src/config/lgln/seg5cls_focal.yaml \
 --load-from /home/yuan/data/out/LGLN/lgln-1973-1975/epoch20.pth  \
 --log-dir /home/yuan/data/out/LGLN --unc-thr 0.2

  PYTHONPATH=. python src/run_lgln.py --mode train \
 --config src/config/lgln/seg5cls_focal.yaml \
 --load-from /home/yuan/data/out/LGLN/lgln-1973-1975/epoch20.pth  \
 --log-dir /home/yuan/data/out/LGLN --unc-thr 0.1


