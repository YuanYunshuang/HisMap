import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from data.utils.vis import overlay_label_on_image, apply_colormap_transparent
from data.bayern import COLORS

def plot_sam_clip_result():
    def load_clip_weight(f, img):
        mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(float)
        mask = 1 - (mask - mask.min()) / (mask.max() - mask.min())
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
        img = apply_colormap_transparent(img, mask, cmap='viridis')
        return img

    def plot_prompt_text(ax, text):
        ax.text(0.5, 0.05, f"prompt: {text}", fontsize=10,
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes,
                bbox=dict(facecolor='w', alpha=0.8, edgecolor='w'))

    sam_in_img1 = cv2.imread("imgs/3821_1974_0_0.png")
    sam_out_img1 = cv2.imread("imgs/3821_1974_0_0_sam.png")
    label1 = cv2.imread("imgs/3821_1974_0_0_label.png", cv2.IMREAD_GRAYSCALE)
    gt_out_img1 = overlay_label_on_image(sam_in_img1, label1, COLORS)

    sam_in_img2 = cv2.imread("imgs/3821_1974_0_5760.png")
    sam_out_img2 = cv2.imread("imgs/3821_1974_0_5760_sam.png")
    label2 = cv2.imread("imgs/3821_1974_0_5760_label.png", cv2.IMREAD_GRAYSCALE)
    gt_out_img2 = overlay_label_on_image(sam_in_img2, label2, COLORS)

    # CLIP
    clip_1a = load_clip_weight("imgs/3821_1974_0_0_[settlement_area_in_a_map].png", sam_in_img1)
    clip_1b = load_clip_weight("imgs/3821_1974_0_0_[buildings_in_a_map].png", sam_in_img1)
    clip_1c = load_clip_weight("imgs/3821_1974_0_0_[area_with_dotted_patterns].png", sam_in_img1)
    clip_2a = load_clip_weight("imgs/3821_1974_0_5760_[woodland_in_a_map].png", sam_in_img2)
    clip_2b = load_clip_weight("imgs/3821_1974_0_5760_[forest_in_a_map].png", sam_in_img2)
    clip_2c = load_clip_weight("imgs/3821_1974_0_5760_[area_with_small_circles].png", sam_in_img2)

    fig = plt.figure(figsize=(18, 6))
    from mpl_toolkits.axes_grid1 import ImageGrid
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, 6),  # creates 2x2 grid of Axes
                     axes_pad=0.1,  # pad between Axes in inch.
                     )
    for ax, img in zip(grid, [sam_in_img1, gt_out_img1, sam_out_img1, clip_1a, clip_1b, clip_1c,
                              sam_in_img2, gt_out_img2, sam_out_img2, clip_2a, clip_2b, clip_2c]):
        img = cv2.resize(img, (512, 512))
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

    fs = 16
    grid[0].set_title("Input Image", fontsize=fs)
    grid[1].set_title("Ground Truth", fontsize=fs)
    grid[2].set_title("SAM", fontsize=fs)
    grid[3].set_title("CLIPSeg a", fontsize=fs)
    grid[4].set_title("CLIPSeg b", fontsize=fs)
    grid[5].set_title("CLIPSeg c", fontsize=fs)

    grid[0].set_ylabel("Settlement", fontsize=fs)
    grid[6].set_ylabel("Woodland", fontsize=fs)


    plot_prompt_text(grid[3], "settlement area in a map")
    plot_prompt_text(grid[4], "building area in a map")
    plot_prompt_text(grid[5], "area with dotted patterns")
    plot_prompt_text(grid[9], "woodland area in a map")
    plot_prompt_text(grid[10], "forest area in a map")
    plot_prompt_text(grid[11], "area with small circles")

    # plt.tight_layout()
    plt.savefig("figs/sam_clip.pdf")
    plt.close()


def label_to_mask(label_file, label):
    L = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
    L = (L == label) * 255
    filename = f"{os.environ.get('HOME')}/Downloads/{os.path.basename(label_file)}"
    cv2.imwrite(filename, L)


def summarize_dataset(file):
    sheets = {'hameln': [], 'bayern': []}
    n_samples = {'hameln': 0, 'bayern': 0}
    with open(file, 'r') as f:
        for line in f:
            area, name = line.strip().split('.')
            map_sheet = name.rsplit("_", 2)[0]
            if map_sheet not in sheets[area]:
                sheets[area].append(map_sheet)
            n_samples[area] += 1

    print(f"Number of map sheets of hameln: {len(sheets['hameln'])}")
    print(f"Number of map sheets of bayern: {len(sheets['bayern'])}")
    print(f"Number of samples of hameln: {n_samples['hameln']}")
    print(f"Number of samples of bayern: {n_samples['bayern']}")


def label_to_rgb(label_file):
    label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
    rgb_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for i in range(1, 5):
        rgb_label[label == i] = COLORS[i-1]
    cv2.imwrite(f"{os.environ.get('HOME')}/Downloads/{os.path.basename(label_file)}",
                rgb_label[..., ::-1])


def crop_img_from_plot(plot_file):
    plot_img = cv2.imread(plot_file)
    for idx in range(5):
        m = idx * 10
        img = plot_img[:, idx*384+m:(idx+1)*384+m]
        cv2.imwrite(f"{os.environ.get('HOME')}/Downloads/crop{idx}.png", img)


def masked_img():
    img = cv2.imread("/home/yuan/Downloads/crop0.png")
    mask = cv2.imread("/home/yuan/Downloads/crop1.png", cv2.IMREAD_GRAYSCALE)
    img = overlay_label_on_image(img, mask.astype(bool), color=COLORS[1][::-1])
    cv2.imwrite(f"{os.environ.get('HOME')}/Downloads/crop_overlay.png", img)




if __name__ == '__main__':
    # plot_sam_clip_result()
    # label_to_mask("/home/yuan/data/HisMap/SMOLdata/hameln/lbls/3821_1974_768_4608.png", 3)
    # summarize_dataset("/home/yuan/data/HisMap/SMOLdata/all.txt")
    label_to_rgb("/home/yuan/data/HisMap/SMOLdata/hameln/lbls/3821_1974_2688_1152.png")
    crop_img_from_plot("/home/yuan/code/HisMap/smolsam/logs/cosine_lr5e-5/2/inf_plots/hameln.3821_1974_2688_1152.png")
    masked_img()