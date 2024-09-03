import copy, shutil
import os, sys
import argparse

import numpy as np

from src.utils.misc import setup_logger
from src.config import load_config, save_config
from src.utils.train_utils import seed_everything
from src.sdl_runner import SupervisedRunner
from src.tools.hameln import generate_4cls_pseudo_labels



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--mode", type=str, default="train",
                        help="train | test")
    parser.add_argument("--resume-from", type=str)
    parser.add_argument("--load-from", type=str)
    parser.add_argument("--log-dir", type=str, default=os.path.join(os.path.dirname(__file__),"../logs"))
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-workers", type=int)
    args = parser.parse_args()

    setup_logger(args.run_name, args.debug)
    seed_everything(2023)
    if args.config is None:
        assert args.load_from is not None or args.resume_from is not None
        if args.load_from is not None:
            args.config = os.path.join(args.load_from, "config.yaml")
        elif args.resume_from is not None:
            args.config = os.path.join(args.resume_from, "config.yaml")
        else:
            raise ValueError("Either --config or --load-from or --resume-from must be specified.")
    cfgs = load_config(args)
    if args.gpus:
        cfgs['TRAIN']['gpus'] = args.gpus
    if args.batch_size is not None:
        cfgs['DATASET']['batch_size'] = args.batch_size
    if args.n_workers is not None:
        cfgs['DATASET']['n_workers'] = args.n_workers
    if args.data_path is not None:
        cfgs['DATASET']['data_path'] = args.data_path

    path = "/media/yuan/luna/data/HisMAP/Hameln"
    exps = ['2002_2005', '1995_1996', '1991_1992', '1985_1986', '1982_1983', '1978', '1974_1975',
            '1960_1961', '1952_1955', '1941_1945', '1936_1937', '1923_1926', '1914_1919', '1897_1898']
    last_years = '2002_2005'
    for i in range(0, len(exps[1:]) - 1):
        years = exps[i]
        next_years = exps[i + 1]
        args.log_dir = args.load_from

        # copy data
        exp_dir = args.log_dir.replace(last_years, years)
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copy(os.path.join(args.log_dir, 'epoch20.pth'), os.path.join(exp_dir, 'epoch0.pth'))

        # train
        args.mode = 'train'
        args.log_dir = exp_dir
        args.config = os.path.join(exp_dir, 'config.yaml')
        args.load_from = exp_dir
        cfgs['TRAIN']['run_name'] = None
        cfgs['TRAIN']['load_from'] = args.load_from
        cfgs['TRAIN']['log_dir'] = args.load_from
        cfgs['DATASET']['gt_dir'] = f'gt_pseudo_{years}'
        train_years = []
        tmp = next_years.split('_')
        train_years.extend(np.arange(int(tmp[0]), int(tmp[1]) + 1).tolist())
        tmp = years.split('_')
        train_years.extend(np.arange(int(tmp[0]), int(tmp[1]) + 1).tolist())
        train_years = [str(x) for x in train_years]
        cfgs['TRAIN']['train_year'] = train_years
        sdl_runner = SupervisedRunner(args, cfgs)
        save_config(cfgs, sdl_runner.runner.logdir)
        sdl_runner.run()

        # test
        args.mode = 'test'
        args.log_dir = args.load_from
        cfgs['DATASET']['gt_dir'] = f'gt_pseudo_{years}'
        cfgs['TEST']['load_from'] = args.load_from
        cfgs['TEST']['log_dir'] = args.load_from
        tmp = next_years.split('_')
        test_years = np.arange(int(tmp[0]), int(tmp[1]) + 1).tolist()
        test_years = [str(x) for x in test_years]
        sdl_runner = SupervisedRunner(args, cfgs)
        sdl_runner.run()

        # generate pseudo labels
        generate_4cls_pseudo_labels(
            f"{path}/exp/seg4cls_{years}/epoch20/holistic_seg_result",
            f"{path}/seg4cls_train/gt_pseudo_{next_years}",
            ['agr', 'for', 'wat', 'set'],
        )

        last_years = years
