import os
import json
import numpy as np
from PIL import Image

color_map = {
    'dynamic': (111, 74, 0),
    'ground': (81, 0, 81),
    'road': (128, 64, 128),
    'sidewalk': (244, 35, 232),
    'parking': (250, 170, 160),
    'rail track': (230, 150, 140),
    'building': (70, 70, 70),
    'wall': (102, 102, 156),
    'fence': (190, 153, 153),
    'guard rail': (180, 165, 180),
    'bridge': (150, 100, 100),
    'tunnel': (150, 120, 90),
    'pole': (153, 153, 153),
    'polegroup': (153, 153, 153),
    'traffic light': (250, 170, 30),
    'traffic sign': (220, 220, 0),
    'vegetation': (107, 142, 35),
    'terrain': (152, 251, 152),
    'sky': (70, 130, 180),
    'person': (220, 20, 60),
    'rider': (255, 0, 0),
    'car': (0, 0, 142),
    'truck': (0, 0, 70),
    'bus': (0, 60, 100),
    'caravan': (0, 0, 90),
    'trailer': (0, 0, 110),
    'train': (0, 80, 100),
    'motorcycle': (0, 0, 230),
    'bicycle': (119, 11, 32),
    'license plate': (0, 0, 142),
}

label_id_map = {
    'dynamic': 5,
    'ground': 6,
    'road': 7,
    'sidewalk': 8,
    'parking': 9,
    'rail track': 10,
    'building': 11,
    'wall': 12,
    'fence': 13,
    'guard rail': 14,
    'bridge': 15,
    'tunnel': 16,
    'pole': 17,
    'polegroup': 18,
    'traffic light': 19,
    'traffic sign': 20,
    'vegetation': 21,
    'terrain': 22,
    'sky': 23,
    'person': 24,
    'rider': 25,
    'car': 26,
    'truck': 27,
    'bus': 28,
    'caravan': 29,
    'trailer': 30,
    'train': 31,
    'motorcycle': 32,
    'bicycle': 33,
    'license plate': -1,
}


kitti_color_palette = [0, 0, 0]  # background color
for class_name in color_map:
    kitti_color_palette.extend(color_map[class_name])
kitti_color_palette.extend([0,] * (255*3 - len(kitti_color_palette)))  # pad the palette to 256 colors
kitti_color_palette.extend([255, 255, 255])  # nonlabeled color


def encode_label(lbl_pil):
    lbl_np = np.array(lbl_pil)
    syntra_lbl_np = np.zeros_like(lbl_np, dtype=np.uint8)
    for class_name, src_id in label_id_map.items():
        if src_id == -1:
            continue
        syntra_lbl_np[lbl_np == src_id] = list(label_id_map.keys()).index(class_name) + 1
    syntra_lbl_pil = Image.fromarray(syntra_lbl_np, mode='P')
    syntra_lbl_pil.putpalette(kitti_color_palette)
    return syntra_lbl_pil


def convert(input_path, output_path):
    split_to_path = {
        'train': os.path.join(input_path, 'training'),
        'test': os.path.join(input_path, 'testing')
    }
    os.makedirs(os.path.join(output_path, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'lbls'), exist_ok=True)
    data_info = {}

    for split, split_path in split_to_path.items():
        img_dir = os.path.join(split_path, 'image_2')
        lbl_dir = os.path.join(split_path, 'semantic')

        cur_samples = []
        for filename in os.listdir(img_dir):
            if filename.endswith('_11.png'):
                continue
            file_id = filename.replace('.png', '')
            img_file = os.path.join(img_dir, filename)
            lbl_file = os.path.join(lbl_dir, filename)

            if not os.path.exists(lbl_file):
                continue

            print(f"Processing {img_file} ...")
            img_pil = Image.open(img_file).convert("RGB")
            lbl_pil = Image.open(lbl_file).convert("L")
            lbl_pil = encode_label(lbl_pil)

            # crop to 3 images
            h, w = img_pil.height, img_pil.width
            wc = w // 3
            for i in range(3):
                img_crop = img_pil.crop((i * wc, 0, (i + 1) * wc, h))
                lbl_crop = lbl_pil.crop((i * wc, 0, (i + 1) * wc, h))
                crop_file_id = f"{file_id}_{i}"
                img_crop.save(os.path.join(output_path, 'imgs', crop_file_id + '.png'))
                lbl_crop.save(os.path.join(output_path, 'lbls', crop_file_id + '.png'))
                cur_samples.append(f"kitti.{crop_file_id}")

                # Process label for data info
                lbl_np = np.array(lbl_crop)
                total_pixels = lbl_np.size
                class_ratios = {}
                for i, class_name in enumerate(label_id_map.keys()):
                    if label_id_map[class_name] == -1:
                        continue
                    class_pixels = np.sum(lbl_np == (i + 1))
                    class_ratios[class_name] = class_pixels / total_pixels
                data_info[f"kitti.{crop_file_id}"] = class_ratios
        
        # save sample list
        with open(os.path.join(output_path, f'{split}_samples.txt'), 'w') as f:
            for sample in cur_samples:
                f.write(sample + '\n')
    # save data info
    with open(os.path.join(output_path, "kitti.json"), 'w') as f:
        json.dump(data_info, f, indent=4)


def save_label_colors(output_folder):
    """Dump the label colors to a json file."""
    with open(os.path.join(output_folder, "color_map.json"), 'w') as f:
        json.dump(color_map, f, indent=4)

if __name__ == "__main__":
    input_path = '/koko/datasets/kitti/data_scene_flow'
    output_path = '/home/yuan/data/syntra/autonomous_driving/kitti'
    # convert(input_path, output_path)
    save_label_colors(output_path)