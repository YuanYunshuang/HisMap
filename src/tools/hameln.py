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


if __name__=="__main__":
    generate_dataset(
        "/home/yuan/data/HisMap/Hameln/TK_projektiv_entzerrt_1m",
        "/home/yuan/data/HisMap/Hameln/Raster_2023",
        "/home/yuan/data/HisMap/Hameln/seg_forest",
        2017,
        'for'
    )
