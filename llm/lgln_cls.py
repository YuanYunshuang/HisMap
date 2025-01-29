import glob
import os
import argparse
import sys

import cv2
import numpy as np

from src.tools.map_analyzer import read_tiff
from src.tools.lgln import LGLN_COLOR

def parse_valid_maps(path_in):
    label_files = glob.glob(os.path.join(path_in, '*/label/*.png'))
    mapfiles = []
    labelfiles = []
    for f in label_files:
        path, basename = os.path.split(f)
        split = path.split('/')[-2]
        patch, year = basename.split('.')[0].split('_')
        label = cv2.imread(f, cv2.IMREAD_GRAYSCALE) / 50
        filename = os.path.join(path_in, split, 'maps', f"{patch}_{year}.tif")
        if not os.path.exists(filename):
            filename = os.path.join(path_in, split, 'label', f"{patch}_{int(year) - 1:d}.tif")
        if not os.path.exists(filename):
            filename = os.path.join(path_in, split, 'label', f"{patch}_{int(year) + 1:d}.tif")
        if not os.path.exists(filename):
            continue
        mapfiles.append(filename)
        labelfiles.append(f)
    return mapfiles, labelfiles


def crop(sheet_file, label_file, out_dir, crop_size, scale=0.5, use_label=False, num_pixel_present=100):
    print(f"Cropping...{sheet_file}")
    out_dir_img = os.path.join(out_dir, 'images', f"{crop_size:d}_{scale:.2f}")
    out_dir_lbl = os.path.join(out_dir, 'labels', f"{crop_size:d}_{scale:.2f}")
    os.makedirs(out_dir_img, exist_ok=True)
    os.makedirs(out_dir_lbl, exist_ok=True)
    TIF = read_tiff(sheet_file)
    TIF = cv2.resize(TIF, None, fx=scale, fy=scale)
    LABEL = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE) / 50
    L_RGB = np.zeros((LABEL.shape[0], LABEL.shape[1], 3), dtype=np.uint8)
    for l in range(1, 6):
        L_RGB[LABEL == l] = LGLN_COLOR[l-1]
    L_RGB = cv2.resize(L_RGB, None, fx=scale, fy=scale)
    crop_size = int(crop_size * scale)
    basename = os.path.basename(sheet_file)
    patch, year = basename.split('.')[0].split('_')

    h, w = TIF.shape[0], TIF.shape[1]
    x = np.arange(0, h, crop_size)
    y = np.arange(0, w, crop_size)
    xs, ys = np.meshgrid(x, y)
    xs, ys = xs.flatten(), ys.flatten()
    for x1, y1 in zip(xs, ys):
        x2, y2 = x1 + crop_size, y1 + crop_size
        x2 = h if x2 > h else x2
        y2 = w if y2 > w else y2
        M = TIF[y1:y2, x1:x2]
        L = L_RGB[y1:y2, x1:x2]
        l_cls = '_'
        if use_label:
            for i in range(1, 5):
                if i == 4 and np.sum(L >= i) > num_pixel_present:
                    l_cls += '1'
                elif np.sum(L == i) > num_pixel_present:
                    l_cls += '1'
                else:
                    l_cls += '0'
        cv2.imwrite(os.path.join(out_dir_img, f"{patch}_{year}_{x1}_{y1}{l_cls}.jpg"), M)
        cv2.imwrite(os.path.join(out_dir_lbl, f"{patch}_{year}_{x1}_{y1}.jpg"), L)


def crop_series(sheet_files, label_files, crop_size, scale=0.5, use_label=False):
    for sf, lf in zip(sheet_files, label_files):
        crop(sf, lf,
             "/home/yuan/data/HisMap/LGLN/dataset/llm_label",
             crop_size, scale, use_label)


if __name__ == '__main__':
    sheet_files, label_files = parse_valid_maps("/home/yuan/data/HisMap/LGLN/dataset/age_tracing_data")
    crop_series(sheet_files, label_files,3072, 0.5, use_label=True)
