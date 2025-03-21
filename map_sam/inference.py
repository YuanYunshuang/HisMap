# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import argparse
import cv2

from map_sam.conditional_sam import mapsam_model_registry, MapPredictor

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


def show_masks(masks):
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((masks.shape[1], masks.shape[2], 4))
    img[:,:,3] = 0
    for m in masks:
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument( "-src", "--source-image", type=str,
        default="/home/yuan/data/HisMap/MapSAM/train_stage0/imgs/3821_1974_0_0.png",
        help="The filename of the source image example", )
    parser.add_argument( "-l", "--source-label", type=str,
        default="/home/yuan/data/HisMap/MapSAM/train_stage0/lbls/3821_1974_0_0.png",
        help="The filename of the label of the source image example", )
    parser.add_argument( "-tgt", "--target-image", type=str,
        default="/home/yuan/data/HisMap/MapSAM/train_stage0/imgs/3821_1974_0_2304.png",
        help="The filename of the target image to be segmented", )
    parser.add_argument("-m", "--model-type", type=str,
        default="vit_b", help="model type", )

    parser.add_argument("-li", "--label-index", type=int, default=3)
    parser.add_argument("-w", "--checkpoint", type=str, default="ckpts/mapsam_model_best.pth")
    parser.add_argument("-o", "--output_dir", type=str, default="./logs")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    return args


def main():
    args = setup_parser()

    mapsam = mapsam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
    mapsam.eval()

    map_predictor = MapPredictor(mapsam)

    # Read src example
    src_img = cv2.imread(args.source_image, cv2.IMREAD_UNCHANGED)
    src_lbl = cv2.imread(args.source_label, cv2.IMREAD_UNCHANGED)
    src_lbl = src_lbl == args.label_index

    map_predictor.set_src_example(src_img, src_lbl)

    # Read tgt image
    tgt_img = cv2.imread(args.target_image, cv2.IMREAD_UNCHANGED)
    masks_np, iou_predictions_np, low_res_masks_np = map_predictor.predict(tgt_img)
    max_iou_mask = masks_np[np.argmax(iou_predictions_np)]

    plt.figure(figsize=(8, 8))
    plt.imshow(src_img)
    show_masks(src_lbl[None])
    plt.axis('off')
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(tgt_img)
    show_masks(max_iou_mask[None])
    plt.axis('off')
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
