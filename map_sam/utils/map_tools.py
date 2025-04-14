import glob

import numpy as np
import matplotlib.pyplot as plt
import os

from torch.fx.experimental.migrate_gradual_types.constraint_transformation import valid_index

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


def overlay_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Overlay a mask on an RGB image.

    Parameters:
    - image: np.ndarray - RGB image (H, W, 3)
    - mask: np.ndarray - Binary or grayscale mask (H, W)
    - color: tuple - RGB color for the mask overlay (default is green)
    - alpha: float - Transparency factor for the overlay (0 to 1)

    Returns:
    - overlayed_image: np.ndarray - Image with mask overlay
    """
    # Ensure image and mask sizes match
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask must have the same height and width")

    # Make sure the image is in RGB (OpenCV loads in BGR by default)
    if image.shape[2] == 3 and np.max(image) > 1.0:
        image = image.astype(np.uint8)

    # Create a color mask
    color_mask = np.zeros_like(image)
    color_mask[mask > 0] = color

    # Blend the images
    overlayed_image = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

    return overlayed_image


def vis_data(mapfile, labelfile, dx=2):
    MAP = read_tiff(mapfile)[::dx, ::dx]
    MAP = cv2.cvtColor(MAP, cv2.COLOR_GRAY2RGB)
    LABEL = read_tiff(labelfile)[::dx, ::dx]
    uniq_idx = np.unique(LABEL)
    valid_index = np.where(LABEL == 1)
    x_min, x_max = np.min(valid_index[0]), np.max(valid_index[0])
    y_min, y_max = np.min(valid_index[1]), np.max(valid_index[1])
    LABEL = LABEL == 1

    # Overlay the mask
    overlay = overlay_mask_on_image(MAP, LABEL, color=(255, 0, 0), alpha=0.3)[x_min:x_max, y_min:y_max]

    cv2.imwrite("/home/yuan/Downloads/tmp.png", overlay)

def process_bayern_data(indir, mapfile, labelfile):
    """
    How to use:
        indir: data root directory with 'lbls' and 'maps' subdirectories.
        mapfile: map file name. Example: '01_489_N_1910'
        labelfile: label file name. Example: '489_1910'
    """
    classes = ['Wald', 'Wiese', 'Siedlung', 'Fluss', 'Stehendes_gewaesser']
    MAP = read_tiff(os.path.join(indir, 'maps', mapfile + '.tif'))

    # get test area info
    ltest = read_tiff(os.path.join(indir, 'lbls', labelfile + '_testsite.tif'))
    valid_index = np.where(ltest == 1)
    x_min, x_max = np.min(valid_index[0]), np.max(valid_index[0])
    y_min, y_max = np.min(valid_index[1]), np.max(valid_index[1])

    # crop test area for the map
    MAP = MAP[x_min:x_max, y_min:y_max]

    # get labels
    LABEL = np.zeros(MAP.shape[:2], dtype=np.uint8)
    for i, c in enumerate(classes):
        lc = read_tiff(os.path.join(indir, 'lbls', f'{labelfile}_{c}.tif'))
        lc = lc[x_min:x_max, y_min:y_max]
        mask = lc == 1
        LABEL[mask] = c

if __name__ == '__main__':
    vis_data(
        "/home/yuan/data/HisMap/Bayern/maps/01_489_N_1910.tif",
        "/home/yuan/data/HisMap/Bayern/label/489_1910_testsite.tif"
    )
    # files = glob.glob('/home/yuan/data/HisMap/Bayern/TK_entzerrt/*.tif')
    # for file in tqdm(files):
    #     convert_tiff_to_png(file, out_dir='/home/yuan/data/HisMap/Bayern/TK_entzerrt_png')