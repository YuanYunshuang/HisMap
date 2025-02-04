import copy, shutil
import glob
import os, sys
import argparse

import numpy as np

from src.utils.misc import setup_logger, save_json, load_json
from src.config import load_config, save_config
from src.utils.train_utils import seed_everything
from src.sdl_runner import SupervisedRunner
from src.tools.hameln import generate_4cls_pseudo_labels, generate_4cls_softpseudo_labels
from src.tools.eval import eval_lgln


def parse_years(years):
    if not '_' in years:
        return [years]
    else:
        tmp = years.split('_')
        years = [str(x) for x in np.arange(int(tmp[0]), int(tmp[1]) + 1).tolist()]
        return years

def update_label_info(labels, meta, years):
    has_new_label = False
    for k, v in labels.items():
        min_year = min(v)
        start_idx = max(meta[k].index(min_year) - 1, 0)
        start_year = meta[k][start_idx]
        if start_year not in v:
            years.append(start_year)
            labels[k].insert(0, start_year)
            has_new_label = True
        max_year = max(v)
        end_idx = min(meta[k].index(max_year) + 1, len(meta[k]) - 1)
        end_year = meta[k][end_idx]
        if end_year not in v:
            years.append(end_year)
            labels[k].append(end_year)
            has_new_label = True
    return labels, years, has_new_label


def load_meta(data_path):
    meta = {}
    for x in sorted(os.listdir(os.path.join(data_path, 'train', 'maps'))):
        patch, year = x.split('.')[0].split('_')
        patch, year = int(patch), int(year)
        if year == 2023:
            continue
        if patch not in meta:
            meta[patch] = []
        meta[patch].append(year)
    for k, v in meta.items():
        meta[k] = sorted(v)
    return meta


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
    parser.add_argument("--unc-thr", type=float, default=0.5)
    parser.add_argument("--reparse-pseudo-label", action="store_true")
    parser.add_argument("--pseudo-label", type=bool, default=True)
    parser.add_argument("--anchor-year", type=int, default=2023)
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
    if args.reparse_pseudo_label:
        cfgs['DATASET']['reparse_pseudo_label'] = args.reparse_pseudo_label
    cfgs['DATASET']['pseudo_label'] = args.pseudo_label

    for hook in cfgs['TRAIN']['hooks']:
        if hook['type'] == 'LGLNPseudoLabelHook':
            hook['unc'] = args.unc_thr

    data_path = cfgs['DATASET']['data_path']
    hook_PL = cfgs['TRAIN']['hooks']
    exp_dir = f"{args.log_dir}/pseudo-{args.unc_thr:.1f}"
    os.makedirs(exp_dir, exist_ok=True)
    load_from = args.load_from
    meta = load_meta(data_path)

    if cfgs['DATASET']['start_year'] == 2023:
        start_year = 2016
        end_year = 2017
        years = [x for x in range(start_year, end_year + 1)]
        labels = {3821: [2016], 3822: [2016], 3921: [2017]}
    else:
        start_year = 1973
        end_year = 1975
        years = [x for x in range(start_year, end_year + 1)]
        labels = {3821: [1974], 3822: [1975], 3921: [1973]}

    json_file = os.path.join(exp_dir, "status.json")
    if os.path.exists(json_file):
        status = load_json(json_file)
        labels = {int(k): v for k, v in status['labels'].items()}
        years = status['years']
        load_from = status['load_from']


    # labels, years, has_new_label = update_label_info(labels, meta, years)
    # years = sorted(list(set(years)))
    # load_from = "/home/yuan/data/out/LGLN/peudo-0.5-0.3/pseudo_11-29-15-57-17"

    while True:
        try:
            # train
            cfgs['TRAIN']['run_name'] = 'pseudo'
            cfgs['TRAIN']['log_dir'] = exp_dir
            cfgs['TRAIN']['load_from'] = load_from
            cfgs['DATASET']['years'] = years
            cfgs['DATASET']['labels'] = labels

            ss = "=======TRAIN=======\n"
            ss += f"LOG_DIR: {exp_dir}\n"
            ss += f"TRAIN_YEARS: {years}\n"
            ss += f"LABEL_INFO:\n"
            for k, v in labels.items():
                ss += f"{k}: {v}\n"
            print(ss)
            sdl_runner = SupervisedRunner(args, cfgs)
            save_config(cfgs, sdl_runner.runner.logdir)
            sdl_runner.run()

            # update config
            labels, years, has_new_label = update_label_info(labels, meta, years)
            if not has_new_label:
                break
            years = sorted(list(set(years)))
            load_from = sdl_runner.runner.logdir
            save_json({'labels': labels, 'years': years, 'load_from': load_from}, os.path.join(exp_dir, "status.json"))
        except Exception as e:
            save_json({'labels': labels, 'years': years, 'load_from': load_from}, os.path.join(exp_dir, "status.json"))
            raise e

    # test
    cfgs['DATASET']['years'] = [1898,1974,1982,1996]
    cfgs['DATASET']['crop_size'] = 1024
    cfgs['DATASET']['crop_step_size'] = 1024
    cfgs['DATASET']['batch_size'] = 1
    cfgs['DATASET']['mode'] = 'test'
    args.load_from = sdl_runner.runner.logdir
    cfgs['TEST']['load_from'] = args.load_from
    args.mode = "test"

    sdl_runner = SupervisedRunner(args, cfgs)
    sdl_runner.run()

    # eval
    eval_lgln(
        f"{sdl_runner.runner.logdir}/holistic_seg_result",
        f"{data_path}/test/label"
    )



