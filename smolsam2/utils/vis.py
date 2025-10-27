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
            lbl_src.float(),
            img_src.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        lbl_src = lbl_src.view(img_src.shape[0], img_src.shape[1], img_src.shape[3], img_src.shape[4])
        T = img_src.shape[1] + 1

        img_src = img_src.detach().permute(0, 1, 3, 4, 2).cpu().numpy() * 0.5 + 0.5
        img_tgt = img_tgt.detach().permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
        lbl_src = lbl_src.detach().cpu().numpy()
        lbl_tgt = lbl_tgt.squeeze(1).detach().cpu().numpy()
        pred_tgt = pred_tgt.detach().sigmoid().squeeze(1).squeeze(1).cpu().numpy()

        for i in range(1):
            midbar = np.zeros_like(img_tgt[0, :, :10]) + 0.5
            imgs = []
            for t in range(T-1):
                imgs.append(img_src[i, t])
                imgs.append(midbar)
                imgs.append(lbl_src[i, t][..., None][..., [0, 0, 0]])
                imgs.append(midbar)
            
            imgs.extend([
                img_tgt[i], midbar,
                pred_tgt[i][..., None][..., [0, 0, 0]], midbar,
                lbl_tgt[i][..., None][..., [0, 0, 0]],
            ])
            try:
                img = np.concatenate(imgs, axis=1)
                img = (img * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(inf_dir, names[i] + '.png'), img)
            except:
                print(f"Failed to save {names[i]}")


def vis_train(img_src, lbl_src, img_tgt, lbl_tgt, pred_tgt, iter_num):
    assert img_src.dim() == 5  # (bs, T, 3, H, W)
    lbl_src = F.interpolate(
            lbl_src.flatten(0, 1).float(),
            img_src.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    bs = min(img_src.shape[0], 8)
    lbl_src = lbl_src.view(img_src.shape[0], img_src.shape[1], img_src.shape[3], img_src.shape[4])
    T = img_src.shape[1] + 1

    ncol = T * 2 + 1
    _, axs = plt.subplots(nrows=bs, ncols=ncol, figsize=(ncol*2, bs*2))

    img_src = img_src.detach().permute(0, 1, 3, 4, 2).cpu().numpy()
    img_tgt = img_tgt.detach().permute(0, 2, 3, 1).cpu().numpy()
    lbl_src = lbl_src.detach().cpu().numpy()
    lbl_tgt = lbl_tgt.squeeze(1).detach().cpu().numpy()
    pred_tgt = pred_tgt.detach().sigmoid().squeeze().cpu().numpy()

    for i in range(bs):
        for t in range(T-1):
            axs[i, t*2].imshow(img_src[i, t, :, :, ::-1])
            axs[i, t*2 + 1].imshow(lbl_src[i, t])
        axs[i, (T-1)*2].imshow(img_tgt[i][..., ::-1])
        axs[i, T*2-1].imshow(lbl_tgt[i])
        axs[i, T*2].imshow(pred_tgt[i])

    plt.tight_layout()
    os.makedirs(f"{os.environ['HOME']}/Downloads/vis_train", exist_ok=True)
    plt.savefig(f"{os.environ['HOME']}/Downloads/vis_train/{iter_num}.jpg")
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


