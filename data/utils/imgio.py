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






