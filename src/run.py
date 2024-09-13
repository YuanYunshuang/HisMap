import copy, shutil
import glob
import os, sys
import argparse

import numpy as np

from src.utils.misc import setup_logger
from src.config import load_config, save_config
from src.utils.train_utils import seed_everything
from src.sdl_runner import SupervisedRunner
from src.tools.hameln import generate_4cls_pseudo_labels


def parse_years(years):
    if not '_' in years:
        return [years]
    else:
        tmp = years.split('_')
        years = [str(x) for x in np.arange(int(tmp[0]), int(tmp[1]) + 1).tolist()]
        return years


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
    exps = [
        # '2016_2017',
        # '2011_2012',
        # '2002_2005',
        # '1995_1996',
        # '1991_1992',
        # '1985_1986',
        # '1982_1983',
        # '1978',
        '1974_1975',  # no 3921
        '1960_1961',
        '1952_1955',
        '1941_1945',
        '1936_1937',  # no 3822
        '1923_1926',  # no 3821
        '1914_1919', # no 3921, 3922
        '1897_1898'
    ]
    ckpt_key = 'all'
    log_dir = '/media/yuan/luna/data/HisMAP/Hameln/exp/seg4cls_'
    max_epoch = 10
    for i in range(len(exps) - 1):
        years = exps[i]
        next_years = exps[i + 1]

        # copy data
        exp_dir = log_dir + years
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copy(os.path.join(log_dir + ckpt_key, 'epoch50.pth'), os.path.join(exp_dir, 'epoch0.pth'))
        if years == '1974_1975':
            files = glob.glob(f'/media/yuan/luna/data/HisMAP/Hameln/seg4cls_train/gt_pseudo_1978/3921*.png')
            for f in files:
                shutil.copy(f, f.replace('1978', '1974_1975'))
        if years == '1936_1937':
            files = glob.glob(f'/media/yuan/luna/data/HisMAP/Hameln/seg4cls_train/gt_pseudo_1941_1945/3822*.png')
            for f in files:
                shutil.copy(f, f.replace('1941_1945', '1936_1937'))
        if years == '1923_1926':
            files = glob.glob(f'/media/yuan/luna/data/HisMAP/Hameln/seg4cls_train/gt_pseudo_1936_1937/3821*.png')
            for f in files:
                shutil.copy(f, f.replace('1936_1937', '1923_1926'))
        if years == '1914_1919':
            files = glob.glob(f'/media/yuan/luna/data/HisMAP/Hameln/seg4cls_train/gt_pseudo_1923_1926/392*.png')
            for f in files:
                shutil.copy(f, f.replace('1923_1926', '1914_1919'))


        # train
        args.mode = 'train'
        args.log_dir = exp_dir
        args.config = os.path.join(exp_dir, 'config.yaml')
        args.load_from = exp_dir
        cfgs['TRAIN']['run_name'] = None
        cfgs['TRAIN']['load_from'] = args.load_from
        cfgs['TRAIN']['log_dir'] = args.load_from
        cfgs['TRAIN']['save_path'] = args.load_from
        cfgs['TRAIN']['max_epoch'] = max_epoch
        cfgs['TRAIN']['lr_scheduler']['milestones'] = [5]
        cfgs['TRAIN']['optimizer']['lr'] = 0.00002
        cfgs['DATASET']['gt_dir'] = f'gt_pseudo_{years}'
        train_years = parse_years(next_years) + parse_years(years)
        cfgs['DATASET']['train_year'] = train_years
        ss = "=======TRAIN=======\n"
        ss += f"LOG_DIR: {exp_dir}\n"
        ss += f"TRAIN_YEARS: {train_years}\n"
        ss += f"GT_DIR: {cfgs['DATASET']['gt_dir']}\n"
        print(ss)
        sdl_runner = SupervisedRunner(args, cfgs)
        save_config(cfgs, sdl_runner.runner.logdir)
        sdl_runner.run()

        # test
        args.mode = 'test'
        args.log_dir = args.load_from
        cfgs['DATASET']['gt_dir'] = f'gt_pseudo_{years}'
        cfgs['TEST']['load_from'] = args.load_from
        cfgs['TEST']['log_dir'] = args.load_from
        test_years = parse_years(next_years)
        cfgs['DATASET']['test_year'] = test_years
        ss = "=======TEST=======\n"
        ss += f"LOG_DIR: {args.log_dir}\n"
        ss += f"TEST_YEARS: {test_years}\n"
        ss += f"GT_DIR: {cfgs['DATASET']['gt_dir']}\n"
        print(ss)
        sdl_runner = SupervisedRunner(args, cfgs)
        sdl_runner.run()

        # generate pseudo labels
        ss = "=======GENERATE PSEUDO LABELS=======\n"
        ss += f"PRED_dir: seg4cls_{years}\n"
        ss += f"OUT_DIR: gt_pseudo_{next_years}"
        print(ss)
        generate_4cls_pseudo_labels(
            f"{path}/exp/seg4cls_{years}/epoch{max_epoch}/holistic_seg_result",
            f"{path}/seg4cls_train/gt_pseudo_{next_years}",
            ['agr', 'for', 'wat', 'set'],
            thr=0.4
        )

        # last_years = years
