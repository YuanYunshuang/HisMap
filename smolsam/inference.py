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
from map_sam.conditional_sam.utils.vis import vis_inference

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
    parser.add_argument( "-l", "--list-file", type=str,
        default="/home/yuan/Downloads/src_tgt.txt",
        help="The filename of the source image example", )
    parser.add_argument( "-d", "--data-root", type=str,
        default="/home/yuan/data/HisMap/MapSAM/train_stage0",
        help="The filename of the source image example", )
    parser.add_argument("-m", "--model-type", type=str,
        default="vit_b", help="model type", )

    parser.add_argument("-li", "--label-index", type=int, default=3)
    parser.add_argument("-c", "--checkpoint", type=str, default="/home/yuan/code/HisMap/map_sam/logs/default-20250321-1433/medsam_model_best.pth")
    parser.add_argument("-o", "--output_dir", type=str, default="./logs")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = setup_parser()

    mapsam = mapsam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
    mapsam.load_state_dict(torch.load(args.checkpoint)['model'])
    mapsam.eval()

    map_predictor = MapPredictor(mapsam)

    src_tgt_list = [x.strip().split(' ') for x in open(args.list_file, "r").readlines()]

    for src, tgt in tqdm(src_tgt_list):
        src_file = os.path.join(args.data_root, 'imgs', src)
        tgt_file = os.path.join(args.data_root, 'imgs', tgt)
        src_label_file = os.path.join(args.data_root, 'lbls', src)

        # Read input
        src_img = cv2.imread(src_file, cv2.IMREAD_UNCHANGED)
        src_lbl_ori = cv2.imread(src_label_file, cv2.IMREAD_UNCHANGED)
        tgt_img = cv2.imread(tgt_file, cv2.IMREAD_UNCHANGED)
        src_lbl = (src_lbl_ori == args.label_index)[::4, ::4]

        src_img_tensor = map_predictor.image_to_tensor(src_img)
        src_lbl_tensor = map_predictor.label_to_tensor(src_lbl)
        tgt_img_tensor = map_predictor.image_to_tensor(tgt_img)

        map_predictor.set_src_example(src_img, src_lbl_ori == args.label_index)
        masks_np, iou_predictions_np, low_res_masks_np = map_predictor.predict(tgt_img)
        map_pred = masks_np[0]

        # map_pred = mapsam(src_img_tensor, src_lbl_tensor.squeeze(1), tgt_img_tensor)[0, 0].cpu().numpy()

        # img_emb = map_predictor.model.image_encoder(torch.cat([src_img_tensor, tgt_img_tensor], dim=0))
        # src_img_emb = img_emb[:1]
        # tgt_img_emb = img_emb[1:]
        # src_img_emb = map_predictor.model.image_encoder(src_img_tensor)
        # prompt_emb_sparse, prompt_emb_dense = map_predictor.model.prompt_encoder(
        #         imgs_emb=src_img_emb,
        #         masks=src_lbl_tensor.float(),
        #     )
        # tgt_img_emb = map_predictor.model.image_encoder(tgt_img_tensor)
        # # Predict masks
        # low_res_masks, iou_predictions = map_predictor.model.mask_decoder(
        #     image_embeddings=tgt_img_emb,
        #     image_pe=map_predictor.model.prompt_encoder.get_dense_pe(),
        #     sparse_prompt_embeddings=prompt_emb_sparse,
        #     dense_prompt_embeddings=prompt_emb_dense,
        #     multimask_output=False,
        # )
        #
        # # Upscale the masks to the original image resolution
        # masks = map_predictor.model.postprocess_masks(low_res_masks, (384, 384))
        # map_pred = masks[0, 0].cpu().numpy()

        vis_inference(
            src_img_tensor.cpu()[0].permute(1, 2, 0).numpy(),
            src_lbl_ori == args.label_index,
            tgt_img_tensor.cpu()[0].permute(1, 2, 0).numpy(),
            map_pred
        )
        pass

if __name__ == "__main__":
    main()
