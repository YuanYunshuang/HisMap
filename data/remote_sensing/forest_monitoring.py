import json
from glob import glob
import os
from PIL import Image
import numpy as np

color_map = {
  "ABBA": [132, 74, 201],
  "ACPE": [14, 191, 102],
  "ACRU": [222, 57, 63],
  "ACSA": [88, 208, 234],
  "Acer": [247, 158, 31],
  "BEAL": [69, 132, 240],
  "BEPA": [185, 92, 38],
  "FAGR": [41, 173, 19],
  "LALA": [236, 57, 174],
  "Mort": [123, 123, 123],
  "Picea": [21, 96, 150],
}

color_palette = [0,] * (255 * 3) + [255, 255, 255]
for i, color in enumerate(color_map.values()):
    color_palette[(i+1) * 3:(i+2)*3] = color


def convert(input_dir, output_dir):
    os.makedirs(os.path.join(output_dir, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'lbls'), exist_ok=True)
    data_info = {}
    cls_id_to_name = {i+1: name for i, name in enumerate(color_map.keys())}

    # dump color map
    color_map_file = os.path.join(output_dir, "color_map.json")
    with open(color_map_file, 'w') as f:
        color_map.update({"background": [0, 0, 0], "nonlabeled": [255, 255, 255]})
        json.dump(color_map, f, indent=4)

    for split in ['train', 'test']:
        cur_samples = []
        input_split_dir = os.path.join(input_dir, split)
        img_files = glob(os.path.join(input_split_dir, "images", "*/*.png"))

        for img_file in img_files:
            print(img_file)
            base_name = os.path.basename(img_file)
            zone = img_file.split('/')[-2]
            label_file = img_file.replace("images", "labels_merged")

            if not os.path.exists(label_file):
                continue

            img = Image.open(img_file).convert("RGB")
            lbl = Image.open(label_file).convert("L")
            lbl.putpalette(color_palette)

            for x in range(img.width // 384):
                for y in range(img.height // 384):
                    img_crop = img.crop((x * 384, y * 384, (x + 1) * 384, (y + 1) * 384))
                    lbl_crop = lbl.crop((x * 384, y * 384, (x + 1) * 384, (y + 1) * 384))
                    crop_name = f"{base_name.replace('.png', '')}_{zone}_{x}_{y}"
                    img_crop.save(os.path.join(output_dir, 'imgs', crop_name + '.png'))
                    lbl_crop.save(os.path.join(output_dir, 'lbls', crop_name + '.png'))
                    cur_samples.append(f"forest_monitoring.trees.{crop_name}")

                    # generate data info
                    lbl_array = np.array(lbl_crop)
                    total_pixels = lbl_array.size
                    class_ratios = {}
                    for cls_id, cls_name in cls_id_to_name.items():
                        class_pixels = np.sum(lbl_array == cls_id)
                        class_ratios[cls_name] = class_pixels / total_pixels
                    data_info[f"forest_monitoring.trees.{crop_name}"] = class_ratios
        # save sample list
        with open(os.path.join(output_dir, f"{split}_samples.txt"), 'w') as f:
            for sample in cur_samples:
                f.write(sample + '\n')
    # save data info
    with open(os.path.join(output_dir, "forest_monitoring.trees.json"), 'w') as f:
        json.dump(data_info, f, indent=4)

def select(output_dir, n=100):
    """Select n samples from the training dataset in each zone."""
    with open(os.path.join(output_dir, "train_samples.txt"), 'r') as f:
        all_samples = [line.strip() for line in f.readlines()]
    
    # summarize class distribution
    with open(os.path.join(output_dir, "forest_monitoring.trees.json"), 'r') as f:
        data_info = json.load(f)
    
    # original distribution
    class_counts = {cls_name: 0 for cls_name in color_map.keys()}
    for sample in all_samples:
        if sample not in data_info:
            print(f"Warning: {sample} not in data info.")
        class_ratios = data_info[sample]
        for cls_name, ratio in class_ratios.items():
            class_counts[cls_name] += ratio
    with open(os.path.join(output_dir, f'train_class_distribution.json'), 'w') as f:
        json.dump(class_counts, f, indent=4)
    
    # class-balanced selection
    selected_samples = []


    cls_to_samples = {k: [] for k in color_map.keys()}
    for sample in all_samples:
        class_ratios = data_info[sample]
        for cls_name, ratio in class_ratios.items():
            cls_to_samples[cls_name].append((sample, ratio))
    for cls_name, samples_ratios in cls_to_samples.items():
        if len(samples_ratios) == 0:
            continue
        selected_samples.extend(np.random.choice(
            [sr[0] for sr in samples_ratios],
            size=max(1, n),
            replace=False
        ).tolist())
    
    # remove repeat samples
    selected_samples = list(set(selected_samples))

    class_counts = {cls_name: 0 for cls_name in color_map.keys()}
    for sample in selected_samples:
        class_ratios = data_info[sample]
        for cls_name, ratio in class_ratios.items():
            class_counts[cls_name] += ratio
    with open(os.path.join(output_dir, f'train_{n}shot_class_distribution.json'), 'w') as f:
        json.dump(class_counts, f, indent=4)
    with open(os.path.join(output_dir, f'train_{n}shot.txt'), 'w') as f:
        for sample in selected_samples:
            f.write(sample + '\n')


if __name__ == "__main__":
    input_directory = "/home/yuan/data/forest_monitoring/tree_data"
    output_directory = "/home/yuan/data/syntra/forest_monitoring/forest_monitoring.trees"
    # convert(input_directory, output_directory)
    select(output_directory, n=100)
