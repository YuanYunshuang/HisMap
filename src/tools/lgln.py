import os
import shutil
from glob import glob
from unittest.mock import patch

import cv2
import numpy as np

from src.tools.map_analyzer import read_tiff

colors = {
    "watery": [[3, 4, 94], [0, 119, 182], [0, 180, 216], [144, 224, 239], [202, 240, 248]],
    "grass": [[191, 210, 0], [212, 215, 0], [221, 223, 0], [238, 239, 32], [255, 255, 63]],
    "mission": [[127, 85, 57], [156, 102, 68], [176, 137, 104], [221, 184, 146], [230, 204, 178]],
    "forest": [[0, 127, 95], [43, 147, 72], [85, 166, 48], [128, 185, 24], [170, 204, 0]]
}

german2english = {'Wald': 'Woodland', 'Grünland': 'Woodland', 'Siedlung': 'Settlement', 'Fließgewässer': 'FlowingWater', 'Stillgewässer': 'StandingWater'}
LGLN_CLS=['Woodland', 'Grassland', 'Settlement', 'FlowingWater', 'StandingWater']

def label_to_rgb(in_file, out_file, color, down_sample=2):
    flow_water = read_tiff(in_file)
    flow_water = flow_water[::down_sample, ::down_sample]
    w, h = flow_water.shape
    flow_water_label = np.zeros((w, h, 3), dtype=np.uint8)
    labels = np.unique(flow_water)
    labels = labels[(labels > 0) & (labels != 3)]
    for i, l in enumerate(labels):
        flow_water_label[flow_water == l] = colors[color][i]
    cv2.imwrite(out_file, flow_water_label[..., ::-1])

def labels_to_rgb(label_path, patch):
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


def crop_pseudo_label(in_path, out_path):
    pass


if __name__ == '__main__':
    path = "/home/yuan/data/HisMap/LGLN/dataset/train/maps"




