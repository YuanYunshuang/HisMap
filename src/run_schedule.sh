PYTHONPATH=. python src/sdl_runner.py --config src/config/segnet/seg4cls_edl.yaml --resume-from logs/seg4cls_edl_07-16-14-43-54
PYTHONPATH=. python src/sdl_runner.py --config src/config/segnet/seg4cls_focal.yaml --run-name seg4cls_focal
PYTHONPATH=. python src/sdl_runner.py --config src/config/segnet/seg4cls_ce.yaml --run-name seg4cls_ce