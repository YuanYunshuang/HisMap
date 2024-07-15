import os
from glob import glob

import cv2
import numpy as np

from src.tools.map_analyzer import read_tiff

MAP_IDS = [3922, 3921, 3822, 3821]
YEARS = [2017, 2011, 2005, 1996, 1992, 1986, 1982, 1978, 1974, 1960, 1955, 1952, 1945, 1936, 1925, 1898]
CATEGORY = ['for']


def generate_dataset(map_dir, raster_dir, out_dir, year, category, img_size=512, overlap=24):
    out_dir_sample = os.path.join(out_dir, 'sample')
    out_dir_gt = os.path.join(out_dir, 'gt')
    os.makedirs(out_dir_sample, exist_ok=True)
    os.makedirs(out_dir_gt, exist_ok=True)
    for id in MAP_IDS:
        print(id)
        mapfiles = glob(os.path.join(map_dir, f'{id}_*.tif'))
        years = np.array([int(os.path.basename(x).split('.')[0].split('_')[1]) for x in mapfiles])
        idx = np.argmin(np.abs(years - year))
        nearest_year = years[idx]
        map_array = read_tiff(mapfiles[idx])
        rasterfile = os.path.join(raster_dir, f'{id}_2023_{category}.tif')
        raster_array = read_tiff(rasterfile)

        H, W = map_array.shape[:2]

        for h in range(0, H, img_size - overlap):
            h_end = min(h + img_size, H)
            for w in range(0, W, img_size - overlap):
                w_end = min(w + img_size, W)
                sample = map_array[h:h_end, w:w_end]
                gt = raster_array[h:h_end, w:w_end]

                if sample.shape[:2] != (img_size, img_size):
                    padh = img_size - sample.shape[0]
                    padw = img_size - sample.shape[1]
                    sample = np.pad(sample, ((0, padh), (0, padw), (0, 0)), mode='constant', constant_values=0)
                    gt = np.pad(gt, ((0, padh), (0, padw)), mode='constant', constant_values=255)
                filename = f"{id}_{nearest_year}_{h}_{w}.png"
                cv2.imwrite(os.path.join(out_dir_sample, filename), sample.astype(np.uint8))
                cv2.imwrite(os.path.join(out_dir_gt, filename), gt.astype(np.uint8))


def extract_color_mask(map_dir, out_dir, year, color):
    out_dir_gt = os.path.join(out_dir, 'corrected_gt')
    os.makedirs(out_dir_gt, exist_ok=True)
    for id in MAP_IDS:
        print(id)
        mapfiles = glob(os.path.join(map_dir, f'{id}_*.tif'))
        years = np.array([int(os.path.basename(x).split('.')[0].split('_')[1]) for x in mapfiles])
        idx = np.argmin(np.abs(years - year))
        nearest_year = years[idx]
        map_array = read_tiff(mapfiles[idx])
        mask = (map_array == np.array(color).reshape(1, 1, 3)).all(axis=-1).astype(np.float32)

        kernel_size = 25
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = np.stack([mask] * 3, axis=-1) * 255

        filename = f"{id}_{nearest_year}.png"
        cv2.imwrite(os.path.join(out_dir_gt, filename), mask)


def tiff_to_png(map_dir):
    for id in MAP_IDS:
        print(id)
        mapfiles = glob(os.path.join(map_dir, f'{id}_*.tif'))
        for mf in mapfiles:
            map_array = read_tiff(mf).astype(np.uint8)
            map_array = np.stack([map_array] * 3, axis=-1)
            cv2.imwrite(mf.replace('tif', 'png'), map_array)


def full_gt_map(map_dir, raster_dir, out_dir, category, colors, year, use_morph=False):
    for id in MAP_IDS:
        print(id)
        mapfiles = glob(os.path.join(map_dir, f'{id}_*.tif'))
        years = np.array([int(os.path.basename(x).split('.')[0].split('_')[1]) for x in mapfiles])
        idx = np.argmin(np.abs(years - year))
        nearest_year = years[idx]
        map_array = read_tiff(mapfiles[idx])
        rasterfile = os.path.join(raster_dir, f'{id}_2023_{category}.tif')
        raster_array = (read_tiff(rasterfile) < 255).astype(float)
        H = min(map_array.shape[0], raster_array.shape[0])
        W = min(map_array.shape[1], raster_array.shape[1])

        mask = np.zeros_like(map_array[..., 0])
        for c in colors:
            mask = np.logical_or(mask, np.abs(map_array - np.array(c).reshape(1, 1, 3)).sum(axis=-1) < 15)
        mask = mask.astype(float)

        if use_morph:
            kernel_size = 25
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        gt_map = np.zeros_like(map_array)
        gt_map[..., 0] = mask * 255
        gt_map[:H, :W, 1] = raster_array[:H, :W] * 255

        filename = f"{id}_{nearest_year}_{category}.png"
        cv2.imwrite(os.path.join(out_dir, filename), gt_map)


if __name__=="__main__":
    # generate_dataset(
    #     "/home/data/HisMAP/Hameln/TK_projektiv_entzerrt_1m",
    #     "/home/data/HisMAP/Hameln/Raster_2023",
    #     "/home/data/HisMAP/Hameln/seg_forest",
    #     2017,
    #     'for'
    # )

    # extract_color_mask(
    #     "/home/data/HisMAP/Hameln/TK_projektiv_entzerrt_1m",
    #     "/home/data/HisMAP/Hameln/seg_forest",
    #     2017,
    #     [198, 211, 137]
    # )

    # tiff_to_png("/home/data/HisMAP/Hameln/Raster_2023")
    color_dict = {
        # 'for': [[198, 211, 137]],
        # 'wat': [[191, 228, 249]],  # [0, 157, 226],
        # 'set': [[247, 187, 164], [226, 0, 14], [211, 211, 210], [116, 115, 106]],
        'agr': [[254, 252, 233], [231, 239, 211]]
    }

    for k, v in color_dict.items():
        full_gt_map(
            "/home/data/HisMAP/Hameln/TK_projektiv_entzerrt_1m",
            "/home/data/HisMAP/Hameln/Raster_2023",
            "/home/data/HisMAP/Hameln/raster_gt",
            k,
            v,
            2017,
            True
        )