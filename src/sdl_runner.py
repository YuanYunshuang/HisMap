import os, sys
import argparse
import logging

import numpy as np
import torch

from src.dataset import get_dataloader
from src.sdl import get_model
from src.utils.misc import setup_logger
from src.config import load_config, save_config
from src.utils.train_utils import seed_everything
from src.runners.train_runner import TrainRunner
from src.runners.test_runner import TestRunner


def ddp_setup():
    from torch.distributed import init_process_group
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class SupervisedRunner:
    def __init__(self, args, cfgs):
        self.mode = args.mode
        if args.gpus > 0:
            self.dist = True
            ddp_setup()
        else:
            self.dist = False
        self.build_runner(args, cfgs)

    def build_runner(self, args, cfgs):
        dataloader = get_dataloader(cfgs['DATASET'],
                                    args.mode,
                                    self.dist)
        model =get_model(cfgs['MODEL'])
        if args.mode == 'train':
            self.runner = TrainRunner(dataloader=dataloader,
                                      model=model,
                                      **cfgs['TRAIN'])
        elif args.mode == 'test':
            self.runner = TestRunner(dataloader=dataloader,
                                     model=model,
                                     **cfgs['TEST'])

    def run(self):
        try:
            self.runner.run()
        finally:
            if self.dist:
                from torch.distributed import destroy_process_group
                destroy_process_group()


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
    parser.add_argument("--year-range", type=str)
    parser.add_argument("--years", type=str)
    parser.add_argument("--crop-size", type=int)
    parser.add_argument("--reparse-pseudo-label", action="store_true")
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
    if args.year_range is not None:
        start_year, end_year = args.year_range.split("-")
        cfgs['DATASET']['start_year'] = int(start_year)
        cfgs['DATASET']['end_year'] = int(end_year)
    if args.years is not None:
        cfgs['DATASET']['years'] = args.years
    if args.crop_size is not None:
        cfgs['DATASET']['crop_size'] = args.crop_size
        if args.mode == "test":
            cfgs['DATASET']['crop_step_size'] = args.crop_size
    if args.reparse_pseudo_label:
        cfgs['DATASET']['reparse_pseudo_label'] = args.reparse_pseudo_label

    sdl_runner = SupervisedRunner(args, cfgs)
    if args.mode == "train":
        save_config(cfgs, sdl_runner.runner.logdir)
    sdl_runner.run()
