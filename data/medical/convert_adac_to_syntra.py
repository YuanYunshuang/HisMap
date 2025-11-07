
import os
from glob import glob
import random
import re
import nibabel as nib
import numpy as np
from PIL import Image
import json

from .utils import med_img_to_rgb_pil_image


semantic_colors = {
    "background": [0, 0, 0],
    "left ventricle": [255, 0, 0],
    "right ventricle": [0, 255, 0],
    "myocardium": [0, 0, 255],
    "non_labeled": [255, 255, 255],
}

semantic_ids = {
    "background": 0,
    "left ventricle": 7,
    "right ventricle": 8,
    "myocardium": 9,
    "non_labeled": 255,
}

palette_colors = {
    0: [0, 0, 0],
    7: [255, 0, 0],
    8: [0, 255, 0],
    9: [0, 0, 255],
    255: [255, 255, 255],
}
pil_palette = np.array([0] * (256 * 3))
for k, v in palette_colors.items():
    pil_palette[k * 3:k * 3 + 3] = v
pil_palette = pil_palette.tolist()



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


def parse_frames_in_folder(folder_path):
    nii_files = glob(os.path.join(folder_path, '*.nii.gz'))
    frame_dict = {}
    for f in nii_files:
        base_name = os.path.basename(f)
        match = re.search(r'frame(\d+).nii.gz', base_name)
        if match:
            frame_num = int(match.group(1))
            frame_dict[frame_num] = f
    return frame_dict

def adac_label_to_syntra_label(lbl):
    syntra_lbl = np.zeros(lbl.shape, dtype=np.uint8)
    for i in range(1, 4):
        syntra_lbl[lbl == i] = i + 6
    # construct pil image
    pil_lbl = Image.fromarray(syntra_lbl, mode='P')
    pil_lbl.putpalette(pil_palette)
    return pil_lbl


def convert(input_folder, output_folder, sample_list_file):
    os.makedirs(os.path.join(output_folder, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'lbls'), exist_ok=True)
    train_samples = []
    for patient in os.listdir(input_folder):
        patient_folder = os.path.join(input_folder, patient)
        frame_dict = parse_frames_in_folder(patient_folder)
        for frame_num, nii_path in frame_dict.items():
            imgs, _, _ = load_nii(nii_path)
            lbls, _, _ = load_nii(nii_path.replace('.nii', '_gt.nii'))
            for i, (img, lbl) in enumerate(zip(imgs.transpose(2,0,1), lbls.transpose(2,0,1))):
                filename = f'{patient}_frame{frame_num:02d}_slice{i:03d}'
                img_pil = med_img_to_rgb_pil_image(img)
                img_pil.save(os.path.join(output_folder, 'imgs', filename + '.png'))
                lbl_pil = adac_label_to_syntra_label(lbl)
                lbl_pil.save(os.path.join(output_folder, 'lbls', filename + '.png'))
                train_samples.append(f"medical.acdc.{filename}")
    # save train.txt
    with open(os.path.join(output_folder, sample_list_file), 'w') as f:
        for sample in train_samples:
            f.write(sample + '\n')


def generate_data_info_file(output_folder):
    """data info file contains the pixel ratios of each class in each image"""
    id_to_semantic = {v: k for k, v in semantic_ids.items()}
    lbl_files = glob(os.path.join(output_folder, 'lbls', '*.png'))
    data_info = {}
    for lbl_file in lbl_files:
        sample_name = 'medical.acdc.' + os.path.basename(lbl_file).replace('.png', '')
        lbl_pil = Image.open(lbl_file)
        lbl_array = np.array(lbl_pil)
        total_pixels = lbl_array.size
        class_ratios = {}
        for class_id in [0, 7, 8, 9, 255]:
            class_pixels = np.sum(lbl_array == class_id)
            class_ratios[id_to_semantic[class_id]] = class_pixels / total_pixels
        data_info[sample_name] = class_ratios
    with open(os.path.join(output_folder, "medical.acdc.json"), 'w') as f:
        json.dump(data_info, f, indent=4)


def dump_label_colors(output_folder):
    """Dump the label colors to a json file."""
    with open(os.path.join(output_folder, "color_map.json"), 'w') as f:
        json.dump(semantic_colors, f, indent=4)


def select_nshot_samples(output_folder, n_list):
    with open(os.path.join(output_folder, "train_samples.txt"), 'r') as f:
        all_samples = [line.strip() for line in f.readlines()]
    for n in n_list:
        selected_samples = random.sample(all_samples, n)
        with open(os.path.join(output_folder, f'train_{n}shot.txt'), 'w') as f:
            for sample in selected_samples:
                f.write(sample + '\n')



if __name__ == '__main__':
    input_folder = "/home/yuan/data/ACDC/testing"
    output_folder = "/home/yuan/data/ACDC/medical.acdc"
    # convert(input_folder, output_folder,"test_samples.txt")
    dump_label_colors(output_folder)
    generate_data_info_file(output_folder)
    select_nshot_samples(output_folder, [100, 50, 10])