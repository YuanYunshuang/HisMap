
import os
from glob import glob
import time
import re
import argparse
import nibabel as nib
import pandas as pd
from medpy.metric.binary import hd, dc
import numpy as np
from PIL import Image



def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header

imgs = load_nii('/home/yuan/data/ADAC/training/patient002/patient002_frame12.nii.gz')[0].transpose(2, 0, 1)
lbls = load_nii('/home/yuan/data/ADAC/training/patient002/patient002_frame12_gt.nii.gz')[0].transpose(2, 0, 1)
 # (T, H, W, D) 4d: (216, 256, 10, frames)

for i, (img, lbl) in enumerate(zip(imgs, lbls)):
    img_pil = Image.fromarray((img/img.max()*255).astype(np.uint8))
    lbl_pil = Image.fromarray((lbl * 80).astype(np.uint8)) 
    img_pil.save(f'/home/yuan/Downloads/images/patient001_frame12_{i}.png')
    lbl_pil.save(f'/home/yuan/Downloads/labels/patient001_frame12_{i}.png')