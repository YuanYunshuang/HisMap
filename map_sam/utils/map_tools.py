import glob

import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import argparse
import cv2
from geotiff import GeoTiff


def read_tiff(filename):
    if not os.path.exists(filename):
        print('File not found: {}'.format(filename))
        return None
    geo_tiff = GeoTiff(filename)
    tiff_array = np.array(geo_tiff.read())
    return tiff_array


def convert_tiff_to_png(tif_file, scale=0.25, out_dir=None):
    step = int(1 / scale)
    TIF = read_tiff(tif_file)
    TIF = TIF[::step,::step]
    if out_dir is None:
        cv2.imwrite(tif_file.replace('.tif', '.png'), TIF)
    else:
        cv2.imwrite(out_dir + '/' + os.path.basename(tif_file).replace('.tif', '.png'), TIF)


if __name__ == '__main__':
    files = glob.glob('/home/yuan/data/HisMap/Bayern/TK_entzerrt/*.tif')
    for file in tqdm(files):
        convert_tiff_to_png(file, out_dir='/home/yuan/data/HisMap/Bayern/TK_entzerrt_png')