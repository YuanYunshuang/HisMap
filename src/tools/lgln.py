import os
import shutil
from glob import glob
from unittest.mock import patch

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.tools.map_analyzer import read_tiff

colors = {
    "watery": [[3, 4, 94], [0, 119, 182], [0, 180, 216], [144, 224, 239], [202, 240, 248]],
    "grass": [[191, 210, 0], [212, 215, 0], [221, 223, 0], [238, 239, 32], [255, 255, 63]],
    "mission": [[127, 85, 57], [156, 102, 68], [176, 137, 104], [221, 184, 146], [230, 204, 178]],
    "forest": [[0, 127, 95], [43, 147, 72], [85, 166, 48], [128, 185, 24], [170, 204, 0]]
}

german2english = {'Wald': 'Woodland', 'Grünland': 'Grassland', 'Siedlung': 'Settlement', 'Fließgewässer': 'FlowingWater', 'Stillgewässer': 'StandingWater'}
LGLN_CLS=['Woodland', 'Grassland', 'Settlement', 'FlowingWater', 'StandingWater']
LGLN_COLOR = [[85, 166, 48], [221, 223, 0], [176, 137, 104], [144, 224, 239], [50, 80, 255]]

def label_to_rgb(in_file, out_file, down_sample=2):
    label = cv2.imread(in_file, cv2.IMREAD_UNCHANGED) / 50
    label = label[::down_sample, ::down_sample]
    w, h = label.shape
    label_rgb = np.zeros((w, h, 3), dtype=np.uint8)
    for l in range(1, 6):
        label_rgb[label == l] = LGLN_COLOR[l-1]
    cv2.imwrite(out_file, label_rgb[..., ::-1])

def gt_to_rgb(label_path, patch):
    label_keys = {'Fließgewässer': 'watery', 'Grünland': 'grass', 'Siedlung': 'mission', 'Wald': 'forest'}
    W, H = 11200, 11500
    rgb_label = np.zeros((W, H, 3), dtype=np.uint8)
    for k, v in label_keys.items():
        filename = os.path.join(label_path, f'{patch}_{k}.tif')
        label = read_tiff(filename)
        mask = label == 1
        inds = np.where(mask)
        rgb_label[inds[0], inds[1]] = colors[v][1]
    cv2.imwrite("/home/yuan/Downloads/label.jpg", rgb_label[..., ::-1])


def tmp():
    flow_water = read_tiff("/home/yuan/data/HisMap/LGLN/TK25_Reference/3922/tif/3922_1898_Fließgewässer.tif")
    flow_water = flow_water[::2, ::2]
    w, h = flow_water.shape
    flow_water_label = np.zeros((w, h, 3), dtype=np.uint8)
    labels = np.unique(flow_water)
    labels = [labels > 0]
    for i, l in enumerate(labels):
        flow_water_label[flow_water == l] = colors['watery'][i]
    cv2.imwrite("/home/yuan/Downloads/tmp.jpg", flow_water_label[..., ::-1])
    green_land = read_tiff("/home/yuan/data/HisMap/LGLN/TK25_Reference/3922/tif/3922_1898_Grünland.tif")
    settlement = read_tiff("/home/yuan/data/HisMap/LGLN/TK25_Reference/3922/tif/3922_1898_Siedlung.tif")
    forest = read_tiff("/home/yuan/data/HisMap/LGLN/TK25_Reference/3922/tif/3922_1898_Wald.tif")


    in_path = "/home/yuan/data/HisMap/LGLN/TK25_Reference/3922/tif"
    out_path = "/home/yuan/data/HisMap/LGLN/dataset/test/gt"
    for german, english in german2english.items():
        files = glob(f"{in_path}/*_{german}.tif")
        for f in files:
            filename = os.path.basename(f)
            filename = filename.replace(german, english)
            shutil.copy(f, f"{out_path}/{filename}")


def remap_label(in_path):
    out_path = in_path.replace('gt', 'label')
    files = sorted(glob(in_path + "/*.tif"))
    data = {p: {} for p in ['3821', '3822', '3921', '3922']}
    for f in files:
        patch, year, cls = os.path.basename(f).split(".")[0].split("_")
        label = LGLN_CLS.index(cls)
        TIF = read_tiff(f)
        if year not in data[patch]:
            data[patch][year] = np.zeros((5,) + TIF.shape, dtype=bool)
        data[patch][year][label] = TIF == 1

    for patch, labels in data.items():
        for year, label in labels.items():
            remapped_label = np.zeros_like(label[0]).astype(np.uint8)
            for i, l in enumerate(label):
                remapped_label[l] = (i + 1) * 50
            cv2.imwrite(os.path.join(out_path, f"{patch}_{year}.png"), remapped_label)


def remap_tk_label(in_path):
    out_path = in_path.replace('gt', 'label')
    os.makedirs(out_path, exist_ok=True)
    years = sorted(os.listdir(in_path))

    for year in years:
        data = {p: {} for p in ['3821', '3822', '3921', '3922']}
        files = sorted(glob(in_path + "/" + year + "/*.tif"))
        for f in files:
            patch, cls = os.path.basename(f).split(".")[0].split("_")
            label = LGLN_CLS.index(german2english[cls])
            TIF = read_tiff(f)
            data[patch][label] = TIF == 1

        for patch, labels in data.items():
            remapped_label = np.zeros_like(labels[0]).astype(np.uint8)
            for i in range(5):
                l = labels[i]
                remapped_label[l] = (i + 1) * 50
            cv2.imwrite(os.path.join(out_path, f"{patch}_{year}.png"), remapped_label)

def crop_sample_map(path):
    map1898 = read_tiff(os.path.join(path, "3821_1898.tif"))
    map1974 = read_tiff(os.path.join(path, "3821_1974.tif"))
    map2016 = read_tiff(os.path.join(path, "3821_2016.tif"))

    x = 3700
    y = 700
    sz = 1600

    cv2.imwrite(os.path.join(path.replace('maps', 'ica2025'), "3821_1898.jpg"), map1898[x:x+sz, y:y+sz])
    cv2.imwrite(os.path.join(path.replace('maps', 'ica2025'), "3821_1974.jpg"), map1974[x:x+sz, y:y+sz, ::-1])
    cv2.imwrite(os.path.join(path.replace('maps', 'ica2025'), "3821_2016.jpg"), map2016[x:x+sz, y:y+sz, ::-1])


def crop_sample_map_and_label_and_prediction():
    map_path = "/home/yuan/data/HisMap/LGLN/dataset/test/maps"
    label_path = "/home/yuan/data/HisMap/LGLN/dataset/test/label"
    pred_path = "/home/yuan/data/out/LGLN/trace/pseudo-0.6/pseudo_12-06-03-57-17/epoch5/holistic_seg_result"
    out_path = "/home/yuan/data/HisMap/LGLN/dataset/icc2025"
    x = 5200
    y = 4100
    sz = 1600
    year = 1996

    hismap = read_tiff(os.path.join(map_path, f"3922_{year}.tif"))[x:x+sz, y:y+sz]
    label = cv2.imread(os.path.join(label_path, f"3922_{year}.png"), cv2.IMREAD_UNCHANGED)[x:x+sz, y:y+sz] / 50
    pred = cv2.imread(os.path.join(pred_path, f"3922_{year}.png"), cv2.IMREAD_UNCHANGED)[x:x+sz, y:y+sz] / 50

    w, h = label.shape
    label_rgb = np.zeros((w, h, 3), dtype=np.uint8)
    pred_rgb = np.zeros((w, h, 3), dtype=np.uint8)
    for l in range(1, 6):
        label_rgb[label == l] = LGLN_COLOR[l-1]
        pred_rgb[pred == l] = LGLN_COLOR[l-1]

    cv2.imwrite(os.path.join(out_path, f"3922_{year}_patch_map.jpg"), hismap[::2, ::2])
    cv2.imwrite(os.path.join(out_path, f"3922_{year}_patch_label.jpg"), label_rgb[::2, ::2, ::-1])
    cv2.imwrite(os.path.join(out_path, f"3922_{year}_patch_pred.jpg"), pred_rgb[::2, ::2, ::-1])


def class_statistic_bar_plot_train():
    label_path = "/home/yuan/data/HisMap/LGLN/dataset/train/label"
    label_files = sorted(glob(os.path.join(label_path, "*.png")))
    label_cnt = np.zeros(6)
    for f in label_files:
        label = cv2.imread(f, cv2.IMREAD_UNCHANGED) / 50
        for l in range(6):
            label_cnt[l] += np.sum(label == l)

    label_cnt /= np.sum(label_cnt)

    label_path = "/home/yuan/data/HisMap/LGLN/dataset/train/label2023"
    label_files = sorted(glob(os.path.join(label_path, "*.png")))
    label_cnt2023 = np.zeros(6)
    for f in label_files:
        label = cv2.imread(f, cv2.IMREAD_UNCHANGED) / 50
        for l in range(6):
            label_cnt2023[l] += np.sum(label == l)

    label_cnt2023 /= np.sum(label_cnt2023)

    bar_width = 0.35
    fontsize = 24
    CLS = ['UK', 'WL', 'GL', 'SM', 'FW', 'SW']
    x = np.arange(len(CLS))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x - bar_width / 2, label_cnt, width=bar_width, label='1973-1975 (bi-directional)')
    ax.bar(x + bar_width / 2, label_cnt2023, width=bar_width, label='2023 (mono-directional)')
    # Add labels and title
    # ax.set_xlabel('Class')
    ax.set_ylabel('Sample ratio (%)', fontsize=fontsize)
    ax.set_title('Sample ratio of train labels', fontsize=fontsize)
    ax.set_xticks(x, CLS, fontsize=fontsize)
    ax.legend(fontsize=fontsize)

    # Show the plot
    plt.tight_layout()
    plt.savefig('/home/yuan/data/HisMap/LGLN/dataset/icc2025/sample_ratio_train.png')
    plt.close()


def class_statistic_bar_plot_test():
    ### TEST data
    label_path = "/home/yuan/data/HisMap/LGLN/dataset/test/label"
    label_files = sorted(glob(os.path.join(label_path, "*.png")))
    bar_width = 0.2
    fontsize = 24
    CLS = ['UK', 'WL', 'GL', 'SM', 'FW', 'SW']
    x = np.arange(len(CLS))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    label_cnt = np.zeros(6)
    for i, f in enumerate(label_files):
        label = cv2.imread(f, cv2.IMREAD_UNCHANGED) / 50
        year = os.path.basename(f)[:-4].split('_')[1]
        for l in range(6):
            label_cnt[l] += np.sum(label == l)

        label_cnt /= np.sum(label_cnt)
        ax.bar(x + bar_width * (i-2), label_cnt, width=bar_width, label=year)
    # Add labels and title
    # ax.set_xlabel('Class')
    ax.set_ylabel('Sample ratio (%)', fontsize=fontsize)
    ax.set_title('Sample ratio of test labels', fontsize=fontsize)
    ax.set_xticks(x, CLS, fontsize=fontsize)
    ax.legend(fontsize=fontsize)

    # Show the plot
    plt.tight_layout()
    plt.savefig('/home/yuan/data/HisMap/LGLN/dataset/icc2025/sample_ratio_test.png')
    plt.close()


if __name__ == '__main__':
    path = "/home/yuan/data/HisMap/LGLN/dataset/train/maps"
    # crop_sample_map_and_label_and_prediction()
    # for f in glob("/home/yuan/data/out/tmp/generate_pred/trace_bi/holistic_seg_result/*.png"):
    #     label_to_rgb(
    #         f,
    #         f.replace(".png", "_rgb_x2_dowmsampled.png"),
    #         2
    #     )
    remap_tk_label("/home/yuan/data/HisMap/LGLN/dataset/age_tracing_data/tk_gt")




