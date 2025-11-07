from math import ceil
import os
import json
import numpy as np
from PIL import Image
from tifffile import imwrite, imread
from tqdm import tqdm


# color map for seismic facies
color_map = {
    "upper_north_sea": [75, 159, 225],
    "middle_north_sea": [87, 199, 133],
    "lower_north_sea": [255, 209, 102],
    "chalk": [242, 142, 43],
    "rijnland": [225, 87, 89],
    "scruff": [175, 122, 161],
    "zechstein": [106, 76, 147]
}

color_palette = [0,] * (255 * 3) + [255, 255, 255]  # background and nonlabeled
for i, color in enumerate(color_map.values()):
    color_palette[(i+1) * 3:(i+2)*3] = color


def float64_to_rg_float32(arr: np.ndarray) -> np.ndarray:
    """
    Encode float64 values into two float32 RGB channels:
    - R holds the high 32 bits (as float32 bit pattern)
    - G holds the low 32 bits (as float32 bit pattern)
    - B is zero
    """
    a = np.asarray(arr, dtype=np.float64)
    u64 = a.view(np.uint64)
    hi = (u64 >> 32).astype(np.uint32)
    lo = (u64 & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    r = hi.view(np.float32)
    g = lo.view(np.float32)
    b = np.zeros_like(r, dtype=np.float32)
    rgb = np.stack([r, g, b], axis=-1)
    return rgb

def save_float64_in_rg_channels_tiff(arr: np.ndarray, out_path: str):
    """
    Save float64 array as RGB float32 TIFF with bit-exact encoding in R and G.
    """
    rgb = float64_to_rg_float32(arr)
    imwrite(out_path, rgb, photometric='rgb')


def load_float64_from_rg_channels_tiff(path: str) -> np.ndarray:
    """
    Decode float64 from an RGB float32 TIFF produced by save_float64_in_rg_channels_tiff.
    """
    rgb = imread(path).astype(np.float32)
    r = rgb[..., 0].view(np.uint32)
    g = rgb[..., 1].view(np.uint32)
    u64 = (r.astype(np.uint64) << 32) | g.astype(np.uint64)
    return u64.view(np.float64)



def process_one_split(facies_file, label_file, output_path, split='train'):
    print(f"Processing {split} split...")
    data_info = {}
    # load data, we convert flaot64 seismic data to float32 grayscale images
    seismic_data = np.load(facies_file).astype(np.float32)
    seismic_label = np.load(label_file).astype(np.float32)
    iline, xline, depth = seismic_label.shape

    # create output dirs
    img_dir = os.path.join(output_path, 'imgs')
    lbl_dir = os.path.join(output_path, 'lbls')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    out_samples = []
    samples = []
    for line in range(iline):
        n = round(xline / 384)
        s = xline // n
        for i in range(n):
            seismic_slice = seismic_data[line, i*s:(i+1)*s, :]
            lbl_slice = seismic_label[line, i*s:(i+1)*s, :]
            samples.append((f"{split}_i{line:03d}_{i}", seismic_slice, lbl_slice))
    for line in range(xline):
        n = round(iline / 384)
        s = xline // n
        for i in range(n):
            seismic_slice = seismic_data[i*s:(i+1)*s, line, :]
            lbl_slice = seismic_label[i*s:(i+1)*s, line, :]
            samples.append((f"{split}_x{line:03d}_{i}", seismic_slice, lbl_slice))
    
    for line_id, seismic_slice, label_slice in tqdm(samples):
        assert seismic_slice.shape == label_slice.shape

        img_pil = Image.fromarray((seismic_slice + 1) / 2, mode='F')
        lbl_pil = Image.fromarray(label_slice.astype(np.uint8), mode='P')
        lbl_pil.putpalette(color_palette)
        slice_id = f'seismic_facies.f3.{line_id}'

        img_pil.save(os.path.join(img_dir, line_id + '.tiff'))
        lbl_pil.save(os.path.join(lbl_dir, line_id + '.png'))
        out_samples.append(slice_id)

        # generate data info
        lbl_array = np.array(lbl_pil)
        total_pixels = lbl_array.size
        class_ratios = {}
        for cls_id, cls_name in enumerate(color_map.keys(), start=1):
            class_pixels = np.sum(lbl_array == cls_id)
            class_ratios[cls_name] = class_pixels / total_pixels
        data_info[slice_id] = class_ratios
    return data_info, out_samples


if __name__ == "__main__":
    
    output_path = '/koko/datasets/syntra/geology/seismic_facies.f3'
    data_info = {}

    # save color_map
    with open(os.path.join(output_path, "color_map.json"), 'w') as f:
        json.dump(color_map, f, indent=4)
    # save meta info
    meta_info = {
        "image_format": "tiff",
        "label_format": "png",
        "image_mode": "F",
        "label_mode": "P",
        "image_channels": 1,
        "num_classes": len(color_map),
    }
    with open(os.path.join(output_path, "meta_info.json"), 'w') as f:
        json.dump(meta_info, f, indent=4)

    # train data
    info, samples = process_one_split(
        "/koko/datasets/sesmic_facies.f3/train/train_seismic.npy",
        "/koko/datasets/sesmic_facies.f3/train/train_labels.npy",
        output_path=output_path, split='train'
    )
    data_info.update(info)
    with open(os.path.join(output_path, 'train_samples.txt'), 'w') as f:
        for sample in samples:
            f.write(sample + '\n')
    
    # test data
    test_samples = []
    info, samples = process_one_split(
        "/koko/datasets/sesmic_facies.f3/test_once/test1_seismic.npy",
        "/koko/datasets/sesmic_facies.f3/test_once/test1_labels.npy",
        output_path=output_path, split='test1'
    )
    data_info.update(info)
    test_samples.extend(samples)
    info, samples = process_one_split(
        "/koko/datasets/sesmic_facies.f3/test_once/test2_seismic.npy",
        "/koko/datasets/sesmic_facies.f3/test_once/test2_labels.npy",
        output_path=output_path, split='test2'
    )
    data_info.update(info)
    test_samples.extend(samples)
    with open(os.path.join(output_path, 'test_samples.txt'), 'w') as f:
        for sample in test_samples:
            f.write(sample + '\n')
    # save data info
    with open(os.path.join(output_path, "seismic_facies.f3.json"), 'w') as f:
        json.dump(data_info, f, indent=4)
