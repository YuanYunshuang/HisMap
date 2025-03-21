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
from geotiff import GeoTiff

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


def read_tiff(filename):
    if not os.path.exists(filename):
        print('File not found: {}'.format(filename))
        return None
    geo_tiff = GeoTiff(filename)
    tiff_array = np.array(geo_tiff.read())
    return tiff_array


def show_masks(masks):
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((masks.shape[1], masks.shape[2], 4))
    img[:,:,3] = 0
    for m in masks:
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def apply_colored_mask(image, mask, color):
    """
    Apply a semi-transparent RGB color mask to an image.

    Parameters:
    - image (np.ndarray): Input image (H, W, 3), dtype=uint8
    - mask (np.ndarray): Binary mask (H, W), values 0 or 1 or boolean
    - color (tuple): RGB color as a tuple, e.g., (255, 0, 0)

    Returns:
    - np.ndarray: Image with color mask applied to masked region
    """
    # Ensure mask is boolean
    mask = mask.astype(bool)

    # Create a copy of the image to overlay on
    output_image = image.copy()

    if mask.any():
        # Create a color layer of the same shape as the image
        color_layer = np.zeros_like(image, dtype=np.uint8)
        color_layer[:, :] = color

        # Alpha blending: 50% transparency
        alpha = 0.5
        output_image[mask] = cv2.addWeighted(image[mask], 1 - alpha, color_layer[mask], alpha, 0)

    return output_image



def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument( "-src", "--source-image", type=str,
        default="/home/yuan/data/HisMap/MapSAM/train_stage0/imgs/3821_1974_0_0.png",
        help="The filename of the source image example", )
    parser.add_argument( "-l", "--source-label", type=str,
        default="/home/yuan/data/HisMap/MapSAM/train_stage0/lbls/3821_1974_0_0.png",
        help="The filename of the label of the source image example", )
    parser.add_argument( "-tgt", "--target-map", type=str,
        default="/home/yuan/data/HisMap/MapSAM/tifs/3821_1974.tif",
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

    # Prepare tgt map
    tgt_map = read_tiff(args.target_map)
    map_h, map_w = tgt_map.shape[:2]
    img_size = src_img.shape[0]
    step_size = img_size // 2
    mask_sheet = np.zeros((map_h, map_w), dtype=bool)

    win = cv2.namedWindow("mask", cv2.WINDOW_AUTOSIZE)
    for x in range(0, map_h, step_size):
        for y in range(0, map_w, step_size):
            tgt_img = tgt_map[x:x+img_size, y:y+img_size]
            masks_np, iou_predictions_np, low_res_masks_np = map_predictor.predict(tgt_img)
            max_iou_mask = masks_np[np.argmax(iou_predictions_np)]
            mask_sheet[x:x+img_size, y:y+img_size] = np.logical_or(mask_sheet[x:x+img_size, y:y+img_size], max_iou_mask)

            map_vis = tgt_map[x:x+img_size:2, max(0, y-2048):y+img_size:2]
            mask_vis = mask_sheet[x:x+img_size:2, max(0, y-2048):y+img_size:2]
            print(f'x, y: {x}, {y}')
            # if y == 10944:
            #     print('d')
            map_vis_merged = apply_colored_mask(map_vis, mask_vis, (0, 255, 0))
            cv2.imshow("mask", map_vis_merged)
            cv2.waitKey(0)
    cv2.destroyWindow("mask")

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
