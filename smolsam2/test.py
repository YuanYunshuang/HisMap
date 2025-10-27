# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import os, sys
import logging

join = os.path.join
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import monai
import argparse


from smolsam2.modeling.dataset import get_dataset
from smolsam2.modeling.mapsam import MapSAM
from smolsam2.modeling.dora import  DoRA_Sam
from smolsam2 import sam_model_registry
from smolsam2.tools.eval import Evaluator
from smolsam2.utils.debug_utils import *

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6



def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--root-dir",
        type=str,
        default="/home/yuan/data/HisMap/SMOL_syntra",
        help="path to training dataset; two subfolders: lbls and imgs",
    )
    parser.add_argument(
        "-m",
        "--model-type",
        type=str,
        default="vit_b",
        help="model type",
    )

    cur_path = os.path.abspath(os.path.dirname(__file__))
    parser.add_argument("-w", "--checkpoint", type=str, default=f"smolsam2/logs/try1/mapsam_model_latest.pth")
    parser.add_argument("-o","--output_dir", type=str)
    parser.add_argument("-r", "--dora-rank", type=int, default=16)
    parser.add_argument("-nw", "--num_workers", type=int, default=4)
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("-s", "--save-result", action="store_true", default=True)
    parser.add_argument("--sam-ckpt", type=str, default=f"{cur_path}/../model_zoo/sam/sam_vit_b_01ec64.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--debug", action="store_true", default=False)
    # dataset
    parser.add_argument("-kw", "--keywords", type=str, default='hameln', help='dataset key words for filtering, e.g., railway, vineyard, bayern, hameln')
    parser.add_argument("--dataset-name", type=str, default='syntra')
    parser.add_argument("--data-info", type=str, default="pairing_train_100shot_test_intercls.json")
    parser.add_argument("--num-src", type=int, default=1, help="number of source images")
    parser.add_argument("-fg", "--foreground-ratio", type=float, default=0.1)
    args = parser.parse_args()

    return args


def eval(pred, target, thresh=0.5):
    assert pred.shape == target.shape
    pred_cls = pred > thresh

    # iou
    intersection = torch.logical_and(pred_cls, target)
    union = torch.logical_or(pred_cls, target)
    iou = torch.sum(intersection) / torch.sum(union)

    # precision
    prec = torch.sum(intersection) / torch.sum(pred_cls)

    # recall
    rec = torch.sum(intersection) / torch.sum(target)

    return iou, prec, rec

def crop_384_to_224(image):
    return image[80:-80, 80:-80]

def crop_96_to_56(image):
    return image[20:-20, 20:-20]

def crop_to_test_area(img_src, lbl_src, img_tgt, lbl_tgt, lbl_pred, mapsam_pred):
    return (img_src[..., 80:-80, 80:-80],
            lbl_src[..., 20:-20, 20:-20],
            img_tgt[..., 80:-80, 80:-80],
            lbl_tgt[..., 80:-80, 80:-80],
            lbl_pred[..., 80:-80, 80:-80],
            mapsam_pred[..., 80:-80, 80:-80])


def single_test(num_notions, args, device, mapsam, seg_loss, ce_loss, logger, out_dir):
    if args.save_result:
        from smolsam2.utils.vis import save_results
    if args.debug:
        from smolsam2.utils.vis import vis_train
    losses = []
    evals = []
    test_dataset = get_dataset(args.dataset_name,
                               data_root=args.root_dir,
                               info_file=args.data_info,
                               num_notions=num_notions,
                               num_src=args.num_src,
                               fg_ratio=args.foreground_ratio,
                               filtering_keys=args.keywords,
                               mode="test")
    num_samples = len(test_dataset)

    logger.info(f"# testing samples: {num_samples}")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=test_dataset.collate_fn,
    )

    evaluator = [Evaluator() for _ in range(num_notions)]

    logger.info(f">>>>>>>>>>>>>>>>>>TESTING {args.keywords}<<<<<<<<<<<<<<<<<<")
    print(f">>>>>>>>>>>>>>>>>>TESTING {args.keywords}<<<<<<<<<<<<<<<<<<")
    with (torch.no_grad()):
        pbar = tqdm(test_dataloader)
        # pbar.set_postfix({'Progress': f'0/{num_samples}', 'Loss': f'{0:.6f}',
        #                   'IoU': f'{0:.3f}', 'Precision': f'{0:.3f}', 'Recall': f'{0:.3f}'})
        for step, data in enumerate(pbar):
            img_src, lbl_src, img_tgt, lbl_tgt = [d.to(device) for d in data[:4]]
            meta = data[4]

            mapsam_pred = mapsam(img_src, lbl_src, img_tgt)
            lbl_pred = mapsam_pred.sigmoid()
            loss = (seg_loss(mapsam_pred, lbl_tgt.unsqueeze(2)) +
                    ce_loss(mapsam_pred, lbl_tgt.float().unsqueeze(2)))
            if "railway" in args.keywords or "vineyard" in args.keywords:
                img_src, lbl_src, img_tgt, lbl_tgt, lbl_pred, mapsam_pred = \
                crop_to_test_area(img_src, lbl_src, img_tgt, lbl_tgt, lbl_pred, mapsam_pred)
            for i in range(num_notions):
                evaluator[i].add_sample(lbl_pred.squeeze(2)[:, i], lbl_tgt[:, i])

            pbar.set_postfix({ 'Loss': f'{loss:.6f}'})
            if args.save_result:
                for i in range(num_notions):
                    save_results(os.path.join(out_dir, f"{str(args.keywords)}_{i+1}"), 
                                img_src, lbl_src[:, :, i], img_tgt, lbl_tgt[:, i], mapsam_pred[:, i], [x['tgt_name'] for x in meta])
            losses.append(loss.item())

    avg_loss = np.array(losses).mean()
    out_info = f"Average Loss: {avg_loss:.6f}\n" + \
        "=========== Evaluation Results ===========\n"
    for i in range(num_notions):
        avg_eval = evaluator[i].eval()
        out_info += \
            f"Notion {i+1}: \n" + \
            f"  IoU (FG): {avg_eval[0]*100:.2f}\n" + \
            f"  Precision (FG): {avg_eval[1]*100:.2f}\n" + \
            f"  Recall (FG): {avg_eval[2]*100:.2f}\n" + \
            f"  Accuracy (BG&FG): {avg_eval[3]*100:.2f}\n" + \
            "------------------------------------------\n"
    logger.info(out_info)
    print(out_info)


def main():
    args = setup_parser()
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        out_dir = os.path.dirname(args.checkpoint)
    device = torch.device(args.device)

    # %% set up model
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("SMOL-Seg")
    logging.basicConfig(filename=join(out_dir, "test.txt"), level=logging.INFO, filemode="a")

    sam_model = sam_model_registry[args.model_type](args.sam_ckpt)
    dora_sam = DoRA_Sam(sam_model, args.dora_rank)
    mapsam = MapSAM(
        image_encoder=dora_sam.sam.image_encoder,
        mask_decoder=dora_sam.sam.mask_decoder,
    ).to(device)
    mapsam.eval()

    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    assert os.path.isfile(args.checkpoint)
    ## Map model to be loaded to specified single GPU
    checkpoint = torch.load(args.checkpoint, map_location=device)
    mapsam.load_dora_parameters(checkpoint["dora-model"])

    # %% test
    num_notions = {'railway': 1, 'vineyard': 1}
    num_notions = {'hameln': 4, 'bayern': 3, 'railway': 1, 'vineyard': 1}
    for kw, n in num_notions.items():
        args.keywords = kw
        single_test(n, args, device, mapsam, seg_loss, ce_loss, logger, out_dir)




if __name__ == "__main__":
    main()
