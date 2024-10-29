import os.path
from glob import glob
import cv2
import numpy as np

from src.tools.map_analyzer import read_tiff


def rescale(in_dir, ratio=0.05):
    out_dir = in_dir + '_thumbnail'
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([x for x in os.listdir(in_dir) if 'png' in x or x.endswith('tif')])

    for file in files:
        print(file)
        if 'tif' in file:
            img = read_tiff(os.path.join(in_dir, file))
        else:
            img = cv2.imread(os.path.join(in_dir, file))
        img = cv2.resize(img, (0, 0), fx = ratio, fy = ratio)
        cv2.imwrite(os.path.join(out_dir, file), img)


def rescale_raster(in_dir):
    out_dir = in_dir + '_thumbnail'
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([x for x in os.listdir(in_dir) if x.endswith('tif')])
    for patch in ['3821_', '3822_', '3921_', '3922_']:
        files = glob(os.path.join(in_dir, f'{patch}2023_*.tif'))
        imgs = []
        categories = []
        sizes = []
        for file in files:
            print(file)
            category = os.path.basename(file).split('.')[0].split('_')[-1]
            categories.append(category)
            img = read_tiff(file)
            img = (img >= 0) & (img < 255)
            img = cv2.resize(img.astype(np.uint8), (0, 0), fx = 0.05, fy = 0.05).astype(bool)
            sizes.append(list(img.shape))
            imgs.append(img)
        max_size = np.array(sizes).max(axis=0)
        img_out = np.zeros((*max_size, 3), dtype=np.uint8)
        for c, I in zip(categories, imgs):
            if c == 'for':
                color = [0, 255, 0]
            elif c == 'set':
                color = [0, 128, 255]
            elif c == 'wat':
                color = [255, 255, 0]
            else:
                color = [50, 200, 200]
            img_out[np.where(I)] = np.array(color)

        cv2.imwrite(os.path.join(out_dir, os.path.join(out_dir, f'{patch}2023.tif')), img_out)


if __name__=="__main__":
    rescale("/home/data/HisMAP/Hameln/TK_projektiv_entzerrt_1m", 0.2)
    # rescale_raster("/home/data/HisMAP/Hameln/Raster_2023")