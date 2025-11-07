import os
import json
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
from PIL import Image
from glob import glob

from utils import med_img_to_rgb_pil_image

label_map = {
    63: 1,   # liver
    126: 2,  # right kidney
    189: 3,  # left kidney
    252: 4,  # spleen
}

color_map = {
    "liver": [255, 128, 128],
    "right_kidney": [128, 255, 128],
    "left_kidney": [64, 64, 255],
    "spleen": [255, 255, 64],
}

color_palette = np.zeros((256 * 3,), dtype=np.uint8)
for idx, color in enumerate(color_map.values(), start=1):
    color_palette[idx * 3:idx * 3 + 3] = color
color_palette[255 * 3:255 * 3 + 3] = [255, 255, 255]  # nonlabeled
color_palette = color_palette.tolist()


def load_dicom(dicom_path):
    """
    Load a DICOM file and return the pixel array and metadata.

    Parameters
    ----------
    dicom_path: string
        Path to the DICOM file.

    Returns
    -------
    arr: np.array
        Numpy array of the pixel data.
    ds: pydicom.Dataset
        The DICOM metadata.
    """
    ds = pydicom.dcmread(dicom_path)          # read metadata + pixel data
    arr = ds.pixel_array                        # numpy array (handles many codecs if pylibjpeg installed)

    # Apply VOI LUT (windowing), if present
    arr = apply_voi_lut(arr, ds)

    return arr, ds


def convert(input_path, output_path):
    """
    Convert a DICOM medical image to Syntra RGB format and save it.

    Parameters
    ----------
    input_path: string
        Path to the chaos dataset which contains two folders: train and test.

    output_path: string
        Path to save the output data.
    """
    # we create the dataset only from MR T2SPIR as it is straghtforward to encode the information in RGB images without loss of quality
    # ct_dir = os.path.join(output_path, 'chaos.mr_t1dual')
    # os.makedirs(os.path.join(ct_dir, 'imgs'), exist_ok=True)
    # os.makedirs(os.path.join(ct_dir, 'lbls'), exist_ok=True)
    mr_dir = os.path.join(output_path, 'chaos.mr_t2spir')
    os.makedirs(os.path.join(mr_dir, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(mr_dir, 'lbls'), exist_ok=True)

    data_info = {}
    class_names = list(color_map.keys())
    for split in ['train', 'test']:
        cur_samples = []
        cur_in_path = os.path.join(input_path, split, "MR")
        dcm_files = glob(os.path.join(cur_in_path, '**/*.dcm'), recursive=True)
        dcm_files = [f for f in dcm_files if 'T2SPIR' in f]

        for dcm_file in dcm_files:
            print(f"Processing {dcm_file}...")
            filename = os.path.basename(dcm_file)
            if "IMG" in filename:
                file_id = filename.split('.')[0]
                lbl_filename = filename
                if 'Phase' in dcm_file:
                    lbl_folder = dcm_file.rsplit('/', 3)[0]
                else:
                    lbl_folder = dcm_file.rsplit('/', 2)[0]
            else:
                file_id = filename.split(',')[0][1:]
                lbl_filename = f'liver_GT_{int(file_id):03d}.png'
                lbl_folder = dcm_file.rsplit('/', 2)[0]
            
            lbl_file = os.path.join(lbl_folder, 'Ground', lbl_filename.replace('.dcm', '.png'))
            if not os.path.exists(lbl_file):
                continue
            img_arr, _ = load_dicom(dcm_file)
            img_pil = med_img_to_rgb_pil_image(img_arr)
            lbl_pil = Image.open(lbl_file)
            lbl_np = np.array(lbl_pil)
            assert all(v in label_map.keys() or v == 0 for v in np.unique(lbl_np)), \
                f"Unexpected label values in {lbl_file}: {np.unique(lbl_np)}"
            total_pixels = lbl_np.size
            class_ratios = {}
            for src_val, tgt_val in label_map.items():
                lbl_np[lbl_np == src_val] = tgt_val
                class_pixels = np.sum(lbl_np == tgt_val) 
                class_ratios[class_names[tgt_val-1]] = class_pixels / total_pixels
            class_ratios["background"] = np.sum(lbl_np == 0) / total_pixels
            data_info[f"chaos.mr_t2spir.{file_id}"] = class_ratios
            
            # same image and label
            lbl_pil = Image.fromarray(lbl_np, mode='P')
            lbl_pil.putpalette(color_palette)
            lbl_pil.save(os.path.join(mr_dir, 'lbls', file_id + '.png'))
            img_pil.save(os.path.join(mr_dir, 'imgs', file_id + '.png'))
            cur_samples.append(f"chaos.mr_t2spir.{file_id}")

        # save sample list
        with open(os.path.join(mr_dir, f'{split}_samples.txt'), 'w') as f:
            for sample in cur_samples:
                f.write(sample + '\n')
    
    # save data info
    info_file = os.path.join(mr_dir, "chaos.mr_t2spir.json")
    with open(info_file, 'w') as f:
        json.dump(data_info, f, indent=4)
    # save color map
    color_map_file = os.path.join(mr_dir, "color_map.json")
    color_map.update({"background": [0, 0, 0], "nonlabeled": [255, 255, 255]})
    with open(color_map_file, 'w') as f:
        json.dump(color_map, f, indent=4)
            





if __name__ == "__main__":
    input_path = '/home/yuan/data/CHAOS'
    output_path = '/home/yuan/data/syntra/medical'
    convert(input_path, output_path)