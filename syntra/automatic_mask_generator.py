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

from smolsam import mapsam_model_registry, MapPredictor
from smolsam.utils.vis import *
from smolsam.utils.map_utils import read_tiff

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


prediction_src_default = {
    1: "0_0",
    # 2: "384_1536",
    # 3: "0_768",
    # 4: "3840_1920",
}

test_sheets = [
    # '3922_1898',
    '3922_1974',
    # '3922_1982',
]


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument( "-src", "--source-image", type=str,
        default="/home/yuan/data/HisMap/SMOLdata/hameln/imgs/3821_1974_384_4608.png",
        help="The filename of the source image example", )
    parser.add_argument( "-tgt", "--target-map", type=str,
        default="/home/yuan/data/HisMap/LGLN/dataset/train/maps/3821_1974.tif",
        help="The filename of the target image to be segmented", )
    parser.add_argument( "-l", "--label-index", type=int,
        default=2,  help="The filename of the label of the source image example", )
    parser.add_argument("-m", "--model-type", type=str,
        default="vit_b", help="model type", )

    parser.add_argument("-w", "--checkpoint", type=str, default="/home/yuan/code/HisMap/smolsam/logs/cosine_lr5e-5/medsam_model_best.pth")
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    return args


def main(args):
    if args.visualize:
        win = cv2.namedWindow("mask", cv2.WINDOW_AUTOSIZE)
    else:
        win = None
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.checkpoint), "map_pred_tmp")
    os.makedirs(args.output_dir, exist_ok=True)

    map_predictor = MapPredictor(args.checkpoint)
    map_predictor.set_src(args.source_image, args.source_image.replace("imgs", "lbls"), args.label_index)

    # Prepare tgt map
    tgt_map = read_tiff(args.target_map)
    if args.label_index > 1:
        tgt_map = tgt_map[..., ::-1]
    if len(tgt_map.shape) == 2:
        tgt_map = tgt_map[:, :, np.newaxis].repeat(3, axis=2)
    sheet_name = os.path.basename(args.target_map).split('.')[0]
    map_h, map_w = tgt_map.shape[:2]
    img_size = map_predictor.src_img.shape[0]
    step_size = 256
    mask_sheet = np.zeros((map_h, map_w), dtype=float)
    total_steps = (map_h // step_size + 1) * (map_w // step_size + 1)

    pbar = tqdm(total=total_steps)

    for x in range(0, map_h, step_size):
        for y in range(0, map_w, step_size):
            pbar.update(1)
            pbar.set_postfix({'Map sheet': sheet_name,  'x, y': f'{x}, {y}'})
            tgt_img = tgt_map[x:x+img_size, y:y+img_size]
            paddingx = img_size - tgt_img.shape[0]
            paddingy = img_size - tgt_img.shape[1]
            tgt_img = np.pad(tgt_img, ((0, paddingx), (0, paddingy), (0, 0)), 'constant', constant_values=0)
            tgt_mask = map_predictor.predict(tgt_img)
            if paddingx > 0 or paddingy > 0:
                tgt_mask = tgt_mask[:img_size-paddingx, :img_size-paddingy]

            mask_sheet[x:x+img_size, y:y+img_size] = np.maximum(mask_sheet[x:x+img_size, y:y+img_size], tgt_mask)
            # mask_sheet[x:x+img_size, y:y+img_size] = (mask_sheet[x:x+img_size, y:y+img_size] + tgt_mask) / 2
            map_vis = tgt_map[x:x+img_size:2, max(0, y-2048):y+img_size:2]
            mask_vis = mask_sheet[x:x+img_size:2, max(0, y-2048):y+img_size:2]

            # if y == 10944:
            #     print('d')

            # vis_inference(map_predictor.src_img, map_predictor.src_lbl, tgt_img, tgt_mask)
            # map_vis_merged = apply_colored_mask(map_vis, mask_vis, (0, 255, 0))
            map_vis_merged = apply_jet_colormap_transparent(map_vis, mask_vis)
            if args.visualize:
                cv2.imshow("mask", map_vis_merged)
                cv2.waitKey(0)
    if args.visualize:
        cv2.destroyWindow("mask")

    cv2.imwrite(os.path.join(args.output_dir, f"{sheet_name}_mask{args.label_index}.png"), (mask_sheet * 255).astype(np.uint8))
    map_vis_merged = apply_jet_colormap_transparent(tgt_map[::4, ::4], mask_sheet[::4, ::4])
    cv2.imwrite(os.path.join(args.output_dir, f"{sheet_name}_mask{args.label_index}_vis.png"), map_vis_merged)

if __name__ == "__main__":
    args = setup_parser()
    path_src = "/home/yuan/data/HisMap/SMOLdata/hameln/imgs"
    path_sheet = "/home/yuan/data/HisMap/LGLN/dataset/test/maps"
    for sheet_name in test_sheets:
        for k, v in prediction_src_default.items():
            args.source_image = join(path_src, f"{sheet_name}_{v}.png")
            args.target_map = join(path_sheet, f"{sheet_name}.tif")
            args.label_index = k
            main(args)
