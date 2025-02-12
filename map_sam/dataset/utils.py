import os

import numpy as np
from geotiff import GeoTiff
import cv2


def read_tiff(filename):
    if not os.path.exists(filename):
        print('File not found: {}'.format(filename))
        return None
    geo_tiff = GeoTiff(filename)
    tiff_array = np.array(geo_tiff.read())
    return tiff_array


def resize_tiff(filename, ratio=1.0):
    img = read_tiff(filename)
    img = cv2.resize(img, None, interpolation=cv2.INTER_AREA, fx=ratio, fy=ratio)
    return img


def resize_tiffs(dirname, ratio=1.0):
    output_dir = dirname + f'_dx_{int(1 /ratio)}'
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(dirname):
        print(filename)
        img = resize_tiff(os.path.join(dirname, filename), ratio=ratio)
        cv2.imwrite(os.path.join(output_dir, filename.replace('tif', 'png')), img)


if __name__ == '__main__':
    resize_tiffs('/home/yuan/data/HisMap/MapSAM/tifs', ratio=0.1)
