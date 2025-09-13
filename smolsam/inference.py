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


from smolsam.modeling.dataset import MapDataset
from smolsam.modeling.mapsam import MapSAM
from smolsam.modeling.dora import  DoRA_Sam
from smolsam import sam_model_registry
from smolsam.tools.eval import Evaluator

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
        default="/home/yuan/data/HisMap/SMOLdata",
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
    parser.add_argument("-w", "--checkpoint", type=str, default=f"/home/yuan/code/HisMap/smolsam/logs/cosine_lr5e-5/medsam_model_best.pth")
    parser.add_argument("-o","--output_dir", type=str)
    parser.add_argument("-r", "--dora-rank", type=int, default=4)
    parser.add_argument("-nw", "--num_workers", type=int, default=4)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-s", "--save-result", action="store_true", default=False)
    parser.add_argument("--sam-ckpt", type=str, default=f"{cur_path}/../model_zoo/sam/sam_vit_b_01ec64.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--debug", action="store_true", default=False)
    # dataset
    parser.add_argument("--data-mode", type=str, default='test')
    parser.add_argument("--data-info", type=str, default='dinfo')
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
    return (img_src[:, :, 80:-80, 80:-80],
            lbl_src[:, 20:-20, 20:-20],
            img_tgt[:, :, 80:-80, 80:-80],
            lbl_tgt[:, 80:-80, 80:-80],
            lbl_pred[:, :, 80:-80, 80:-80],
            mapsam_pred[:, :, 80:-80, 80:-80])


def single_test(label, args, device, mapsam, seg_loss, ce_loss, logger, out_dir):
    if args.save_result:
        from smolsam.utils.vis import save_results
    if args.debug:
        from smolsam.utils.vis import vis_train
    losses = []
    evals = []
    test_dataset = MapDataset(args.root_dir,
                              mode=args.data_mode,
                              info_file=args.data_info,
                              fix_src_lbl=label,
                              fg_ratio=args.foreground_ratio)
    num_samples = len(test_dataset)

    logger.info(f"# testing samples: {num_samples}")
    train_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    evaluator = Evaluator()

    logger.info(f">>>>>>>>>>>>>>>>>>TESTING LABEL {label}<<<<<<<<<<<<<<<<<<")
    print(f">>>>>>>>>>>>>>>>>>TESTING LABEL {label}<<<<<<<<<<<<<<<<<<")
    with (torch.no_grad()):
        pbar = tqdm(train_dataloader)
        # pbar.set_postfix({'Progress': f'0/{num_samples}', 'Loss': f'{0:.6f}',
        #                   'IoU': f'{0:.3f}', 'Precision': f'{0:.3f}', 'Recall': f'{0:.3f}'})
        for step, data in enumerate(pbar):
            img_src, lbl_src, img_tgt, lbl_tgt = [d.to(device) for d in data[:4]]
            meta = data[4]

            mapsam_pred = mapsam(img_src, lbl_src, img_tgt)
            lbl_pred = mapsam_pred.sigmoid()
            loss = (seg_loss(mapsam_pred, lbl_tgt.unsqueeze(1)) +
                    ce_loss(mapsam_pred, lbl_tgt.float().unsqueeze(1)))
            if "siegfried" in args.data_mode:
                img_src, lbl_src, img_tgt, lbl_tgt, lbl_pred, mapsam_pred = \
                crop_to_test_area(img_src, lbl_src, img_tgt, lbl_tgt, lbl_pred, mapsam_pred)
            evaluator.add_sample(lbl_pred.squeeze(1), lbl_tgt)

            pbar.set_postfix({ 'Loss': f'{loss:.6f}'})
            if args.save_result:
                save_results(os.path.join(out_dir, f"{args.data_mode}_{str(label)}"), img_src, lbl_src, img_tgt, lbl_tgt, mapsam_pred, meta['tgt_name'])
            if args.debug and (step + 1) % 50 == 0:
                vis_train(img_src, lbl_src, img_tgt, lbl_tgt, mapsam_pred)
            losses.append(loss.item())

    avg_loss = np.array(losses).mean()
    avg_eval = evaluator.eval()
    out_info = \
        f"Loss: {avg_loss:.6f}\n" + \
        f"IoU (FG): {avg_eval[0]*100:.2f}\n" + \
        f"Precision (FG): {avg_eval[1]*100:.2f}\n" + \
        f"Recall (FG): {avg_eval[2]*100:.2f}\n" + \
        f"Accuracy (BG&FG): {avg_eval[3]*100:.2f}\n"
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
    mapsam.load_dora_parameters(checkpoint["model"])

    # %% test
    if 'siegfried_railway' in args.data_mode:
        labels = [1]
    elif 'siegfried_vineyard' in args.data_mode:
        labels = [2]
    elif 'bayern' in args.data_mode:
        labels = [1, 2, 4]
    else:
        labels = [1, 2, 3, 4]
    for label in labels:
        single_test(label, args, device, mapsam, seg_loss, ce_loss, logger, out_dir)




if __name__ == "__main__":
    main()
