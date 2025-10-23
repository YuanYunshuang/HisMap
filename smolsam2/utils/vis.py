import os

import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np
import cv2
import torch

def save_results(out_dir, img_src, lbl_src, img_tgt, lbl_tgt, pred_tgt, names, label_only=False):
    bs = pred_tgt.shape[0]

    if label_only:
        inf_dir = os.path.join(out_dir, f'inf_labels')
        os.makedirs(inf_dir, exist_ok=True)
        pred_tgt = pred_tgt.detach().sigmoid().squeeze().cpu().numpy()
        pred_tgt = (pred_tgt * 255).astype(np.uint8)
        for i in range(bs):
            cv2.imwrite(os.path.join(inf_dir, names[i] + '.png'), pred_tgt[i])
    else:
        inf_dir = os.path.join(out_dir, f'inf_plots')
        os.makedirs(inf_dir, exist_ok=True)
        lbl_src = F.interpolate(
            lbl_src.unsqueeze(1).float(),
            img_src.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        img_src = img_src.detach().permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
        img_tgt = img_tgt.detach().permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
        lbl_src = lbl_src.detach().cpu().numpy()
        lbl_tgt = lbl_tgt.detach().cpu().numpy()
        pred_tgt = pred_tgt.detach().sigmoid().squeeze().cpu().numpy()

        for i in range(bs):
            midbar = np.zeros_like(img_src[0, :, :10]) + 0.5
            img = np.concatenate((
                img_src[i], midbar,
                lbl_src[i][..., None][..., [0, 0, 0]], midbar,
                img_tgt[i], midbar,
                pred_tgt[i][..., None][..., [0, 0, 0]], midbar,
                lbl_tgt[i][..., None][..., [0, 0, 0]],
            ), axis=1)
            img = (img * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(inf_dir, names[i] + '.png'), img)


def vis_train(img_src, lbl_src, img_tgt, lbl_tgt, pred_tgt):
    assert img_src.dim() == 5 and img_src.shape[1] == 1, "Only support one src image visualization."
    lbl_src = F.interpolate(
            lbl_src.float(),
            img_src.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    bs = img_src.shape[0]

    fig, axs = plt.subplots(nrows=bs, ncols=5, figsize=(10, bs*2))

    img_src = img_src.squeeze(1).detach().permute(0, 2, 3, 1).cpu().numpy()
    img_tgt = img_tgt.detach().permute(0, 2, 3, 1).cpu().numpy()
    lbl_src = lbl_src.detach().cpu().numpy()
    lbl_tgt = lbl_tgt.detach().cpu().numpy()
    pred_tgt = pred_tgt.squeeze(1).detach().sigmoid().squeeze().cpu().numpy()

    for i in range(bs):
        axs[i, 0].imshow(img_src[i][..., ::-1])
        axs[i, 1].imshow(lbl_src[i])
        axs[i, 2].imshow(img_tgt[i][..., ::-1])
        axs[i, 3].imshow(lbl_tgt[i])
        axs[i, 4].imshow(pred_tgt[i])

    plt.tight_layout()
    plt.savefig("/home/yuan/Downloads/vis_train.jpg")
    plt.close()


def vis_inference(img_src, lbl_src, img_tgt, pred_tgt):
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(8, 2))

    axs[0].imshow(img_src[..., ::-1])
    axs[1].imshow(lbl_src)
    axs[2].imshow(img_tgt[..., ::-1])
    axs[3].imshow(pred_tgt)

    plt.tight_layout()
    plt.savefig("/home/yuan/Downloads/vis_inference.png")
    plt.close()


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
    output_image = image[..., ::-1].copy()

    if mask.any():
        # Create a color layer of the same shape as the image
        color_layer = np.zeros_like(image, dtype=np.uint8)
        color_layer[:, :] = color

        # Alpha blending: 50% transparency
        alpha = 0.5
        output_image[mask] = cv2.addWeighted(image[mask], 1 - alpha, color_layer[mask], alpha, 0)

    return output_image


def apply_jet_colormap_transparent(image, mask):
    """
    Apply a transparent Jet colormap to an image using a float mask.

    Parameters:
    - image (np.ndarray): Input image (H, W, 3), RGB, dtype=uint8
    - mask (np.ndarray): Float mask (H, W), values in [0.0, 1.0]

    Returns:
    - np.ndarray: Image with Jet colormap blended over original image
    """
    # Ensure mask is float32 and clipped between 0 and 1
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)

    # Resize if needed
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask must have the same height and width")

    # Convert mask to Jet colormap using matplotlib
    jet_map = plt.get_cmap('jet')
    jet_colors = jet_map(1 - mask)[..., :3] # Ignore alpha, get RGB only
    # jet_colors[..., -1] = 0.6
    jet_colors = (jet_colors * 255).astype(np.uint8)

    # Blend the jet colormap with the original image using the mask as alpha
    blended = image.copy().astype(np.float32)
    jet_colors = jet_colors.astype(np.float32)

    for c in range(3):
        blended[:, :, c] = (1 - 0.5) * blended[:, :, c] + 0.5 * jet_colors[:, :, c]

    return blended.astype(np.uint8)


