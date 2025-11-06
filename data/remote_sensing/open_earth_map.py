import os
from PIL import Image
import numpy as np
from glob import glob
import json


semantic_colors = {
    "background": [0, 0, 0], #black
    "tree": [0, 255, 128], #spring green
    "rangeland": [244, 164, 96], #sandy brown
    "bareland": [210, 180, 140], #aqua brown
    "argric_land_type1": [255, 255, 0], #yellow
    "road_type1": [128, 64, 128], #purple
    "sea_lake_pond": [30, 144, 255], #dodger blue
    "building_type1": [255, 0, 128], #deep pink
    "road_type2": [0, 128, 255], #blueish
    "river": [0, 255, 255], #cyan
    "boat_ship": [255, 128, 0], #orange
    "agric_land_type2": [128, 255, 0], #lime green
    "vehicle_cargo_trailer": [128, 0, 255], #violet
    "parking_spaces": [128, 192, 128], #grayish green
    "sports_field": [192, 192, 192], #silver
    "building_type2": [255, 0, 100], #hot pink
    "developed_space": [50, 128, 50], #dark green
    "non_labeled": [255, 255, 255], #white
}

semantic_ids = {
    "background": 0,
    "tree": 10,
    "rangeland": 11,
    "bareland": 12,
    "argric_land_type1": 13,
    "road_type1": 14,
    "sea_lake_pond": 15,
    "building_type1": 16,
    "road_type2": 17,
    "river": 18,
    "boat_ship": 19,
    "agric_land_type2": 20,
    "vehicle_cargo_trailer": 21,
    "parking_spaces": 22,
    "sports_field": 23,
    "building_type2": 24,
    "developed_space": 25,
    "non_labeled": 255,
}

palette_colors =  {cls_id: semantic_colors[cls_name] for cls_name, cls_id in semantic_ids.items()}
pil_palette = [0] * (256 * 3)
for k, v in palette_colors.items():
    pil_palette[k * 3:k * 3 + 3] = v
pil_palette = pil_palette

base_cls_id_to_syntra_id = {
    1: 10,  # tree
    2: 11,  # rangeland
    3: 12,  # bareland
    4: 13,  # argric_land_type1
    5: 14,  # road_type1
    6: 15,  # sea_lake_pond
    7: 16,  # building_type1
}

val_novel_cls_id_to_syntra_id = {
    8: 17,  # road_type2
    9: 18,  # river
    10: 19, # boat_ship
    11: 20, # agric_land_type2
}

test_novel_cls_id_to_syntra_id = {
    8: 21,  # vehicle_cargo_trailer
    9: 22,  # parking_spaces
    10: 23, # sports_field
    11: 24, # building_type2
}

oem_landcover_cls_id_to_syntra_id = {
    1: 12,  # bareland
    2: 11,  # rangeland
    3: 25,  # developed_space
    4: 14,  # road_type1
    5: 10,  # tree
    6: 15,  # sea_lake_pond
    7: 13,  # argric_land_type1
    8: 16,  # building_type1
}


def convert_label(lbl, split):
    """
    Convert the original label to syntra format.
    """
    lbl = np.array(lbl)
    lbl_converted = np.zeros_like(lbl) 
    if split == 'val':
        cls_id_to_syntra_id = {**base_cls_id_to_syntra_id, **val_novel_cls_id_to_syntra_id}
    elif split == 'test':
        cls_id_to_syntra_id = {**base_cls_id_to_syntra_id, **test_novel_cls_id_to_syntra_id}
    elif split == 'landcover':
        cls_id_to_syntra_id

    for orig_id, syntra_id in oem_landcover_cls_id_to_syntra_id.items():
        lbl_converted[lbl == orig_id] = syntra_id
    
    lbl_converted = Image.fromarray(lbl_converted.astype(np.uint8), mode='P')
    lbl_converted.putpalette(pil_palette)
    return lbl_converted


def convert(img_path, lbl_path, split):
    filename = os.path.basename(img_path)
    img = Image.open(img_path).convert('RGB')
    if img.size[0] != img.size[1]:
        print(f'Warning: image {filename} is not square: {img.size}')
    new_size = round(img.size[0] / 384.0) * 384
    print(f'Resizing image {filename} from {img.size} to ({new_size}, {new_size})')
    img = img.resize((new_size, new_size), resample=Image.BILINEAR)
    
    if os.path.exists(lbl_path):
        lbl = Image.open(lbl_path).convert('P')
        # convert label
        lbl = convert_label(lbl, split)
        lbl = lbl.resize((new_size, new_size), resample=Image.NEAREST)
    else:
        lbl = None
    return img, lbl, new_size


def convert_landcover(input_path, output_path):
    """
    Convert OpenEarthMap dataset to syntra format.
    """
    cities = [x for x in os.listdir(os.path.join(input_path)) if os.path.isdir(os.path.join(input_path, x))]

    for city in cities:
        # ignore images without labels
        filenames = os.listdir(os.path.join(input_path, city, 'labels'))
        cur_out_path = os.path.join(output_path, f'oem.landcover.{city}')
        if os.path.exists(cur_out_path):
            continue
        os.makedirs(os.path.join(cur_out_path, 'imgs'), exist_ok=True)
        os.makedirs(os.path.join(cur_out_path, 'lbls'), exist_ok=True)
        for filename in filenames:
            img_path = os.path.join(input_path, city, 'images', filename)
            lbl_path = os.path.join(input_path, city, 'labels', filename)
            if not os.path.exists(img_path):
                print(f'Warning: image {filename} does not exist, skipping.')
                continue
            img, lbl, new_size = convert(img_path, lbl_path, city)

            # crop to 384x384 patches
            n = new_size // 384
            for i in range(n):
                for j in range(n):
                    img_patch = img.crop((j*384, i*384, (j+1)*384, (i+1)*384))
                    lbl_patch = lbl.crop((j*384, i*384, (j+1)*384, (i+1)*384))

                    patch_filename = filename.replace('.tif', f'_{i}_{j}.png')
                    img_patch.save(os.path.join(cur_out_path, 'imgs', patch_filename))
                    lbl_patch.save(os.path.join(cur_out_path, 'lbls', patch_filename))


def convert_fewshot(input_path, output_path):
    """
    Convert OpenEarthMap dataset to syntra format.
    """
    splits = ['train', 'val', 'test']
    output_path = os.path.join(output_path, f'oem.fewshot')
    os.makedirs(os.path.join(output_path, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'lbls'), exist_ok=True)

    for split in splits:
        cur_input_path = os.path.join(input_path, split + 'set')
        sample_list = {'support_set': [], 'query_set': []}
        print(f'Processing split: {split}')
        filenames = os.listdir(os.path.join(cur_input_path, 'images'))
        for filename in filenames:
            img_path = os.path.join(cur_input_path, 'images', filename)
            lbl_path = os.path.join(cur_input_path, 'labels', filename)

            img, lbl, new_size = convert(img_path, lbl_path, split)

            # crop to 384x384 patches
            for i in range(new_size // 384):
                for j in range(new_size // 384):
                    patch_filename = filename.replace('.tif', f'_{i}_{j}.png')
                    img_patch = img.crop((j*384, i*384, (j+1)*384, (i+1)*384))
                    img_patch.save(os.path.join(output_path, 'imgs', patch_filename))
                    if lbl is not None:
                        lbl_patch = lbl.crop((j*384, i*384, (j+1)*384, (i+1)*384))
                        lbl_patch.save(os.path.join(output_path, 'lbls', patch_filename))
                        sample_list['support_set'].append("oem.fewshot." + patch_filename[:-4])
                    else:
                        sample_list['query_set'].append("oem.fewshot." + patch_filename[:-4])

        if split == 'train':
            split_file = 'train_samples.txt'
            with open(os.path.join(output_path, split_file), 'w') as f:
                for sample in sample_list['support_set']:
                    f.write(sample + '\n')
        else:
            for set_name, samples in sample_list.items():
                split_file = f'{split}_{set_name}.txt'
                with open(os.path.join(output_path, split_file), 'w') as f:
                    for sample in samples:
                        f.write(sample + '\n')


def generate_data_info_file(output_folder):
    """data info file contains the pixel ratios of each class in each image"""
    id_to_semantic = {v: k for k, v in semantic_ids.items()}
    lbl_files = glob(os.path.join(output_folder, 'lbls', '*.png'))
    data_info = {}
    for lbl_file in lbl_files:
        sample_name = 'oem.fewshot.' + os.path.basename(lbl_file).replace('.png', '')
        lbl_pil = Image.open(lbl_file)
        lbl_array = np.array(lbl_pil)
        total_pixels = lbl_array.size
        class_ratios = {}
        for class_id in list(range(10, 25)) + [0, 255]:
            class_pixels = np.sum(lbl_array == class_id)
            class_ratios[id_to_semantic[class_id]] = class_pixels / total_pixels
        data_info[sample_name] = class_ratios
    with open(os.path.join(output_folder, "oem.fewshot.json"), 'w') as f:
        json.dump(data_info, f, indent=4)


def dump_label_colors_fewshot(output_folder):
    """Dump the label colors to a json file."""
    selected = list(base_cls_id_to_syntra_id.values()) + \
               list(val_novel_cls_id_to_syntra_id.values()) + \
               list(test_novel_cls_id_to_syntra_id.values()) + [0, 255]
    color_map = {k: v for k, v in semantic_colors.items() if semantic_ids[k] in selected}
    with open(os.path.join(output_folder, "color_map.json"), 'w') as f:
        json.dump(color_map, f, indent=4)


def dump_label_colors_landcover(output_folder):
    """Dump the label colors to a json file."""
    selected = list(oem_landcover_cls_id_to_syntra_id.values()) + [0, 255]
    color_map = {k: v for k, v in semantic_colors.items() if semantic_ids[k] in selected}
    with open(os.path.join(output_folder, "color_map.json"), 'w') as f:
        json.dump(color_map, f, indent=4)


def select_from_oem_landcover(output_path, num_per_city=20):
    """
    Select a few images from each city in the OpenEarthMap landcover dataset.
    """
    cities = [x for x in os.listdir(os.path.join(output_path)) if x.startswith('oem.landcover.')]
    selected_output_path = os.path.join(output_path, 'oem.landcover_selection')
    os.makedirs(os.path.join(selected_output_path, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(selected_output_path, 'lbls'), exist_ok=True)
    data_info = {}

    for city in cities:
        with open(os.path.join(output_path, city, 'oem.fewshot.json'), 'r') as f:
            city_data_info = json.load(f)
        labled_files = [x for x in os.listdir(os.path.join(output_path, city, 'lbls'))]
        labled_files.sort()
        img_files = [os.path.join(output_path, city, 'imgs', x) for x in labled_files]
        lbl_files = [os.path.join(output_path, city, 'lbls', x) for x in labled_files]
        if len(img_files) == 0:
            print(f'Warning: city {city} has no images. Skipping.')
            continue
        # randomly select num_per_city images
        selection_inds = np.random.choice(len(img_files), min(num_per_city, len(img_files)), replace=False).tolist()
        selected_imgs = [img_files[i] for i in selection_inds]
        selected_lbls = [lbl_files[i] for i in selection_inds]
        for img_file, lbl_file in zip(selected_imgs, selected_lbls):
            filename = os.path.basename(img_file)
            data_info['oem.landcover_selection.' + filename.replace('.png', '')] = city_data_info['oem.fewshot.' + filename.replace('.png', '')]
            # copy image and label 
            os.system(f'cp {img_file} {os.path.join(selected_output_path, "imgs", filename)}')
            os.system(f'cp {lbl_file} {os.path.join(selected_output_path, "lbls", filename)}')
    with open(os.path.join(selected_output_path, "oem.landcover_selection.json"), 'w') as f:
        json.dump(data_info, f, indent=4)


def split_selection(selected_output_path):
    with open(os.path.join(selected_output_path, 'oem.landcover_selection.json'), 'r') as f:
        data_info = json.load(f)
    filenames = list(data_info.keys())
    np.random.shuffle(filenames)
    num_samples = len(filenames)
    train_files = filenames[:num_samples * 4 // 10]
    val_files = filenames[num_samples * 4 // 10:num_samples * 6 // 10]
    test_files = filenames[num_samples * 6 // 10:]
    for split, split_files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
        with open(os.path.join(selected_output_path, f'{split}_samples.txt'), 'w') as f:
            for sample in split_files:
                f.write(sample + '\n')   

    

if __name__ == "__main__":
    # input_path = '/home/yuan/data/OpenEarthMap_fewshot'
    # output_path = '/home/yuan/data/syntra/remote_sensing'
    # convert_fewshot(input_path, output_path)
    # dump_label_colors_fewshot(os.path.join(output_path, 'oem.fewshot'))
    # generate_data_info_file(os.path.join(output_path, 'oem.fewshot'))

    # input_path = '/home/yuan/data/OpenEarthMap/OpenEarthMap_wo_xBD'
    # output_path = '/home/yuan/data/syntra/remote_sensing'
    # convert_landcover(input_path, output_path)
    # for city_folder in os.listdir(os.path.join(output_path)):
    #     if 'oem.landcover' not in city_folder:
    #         continue
    #     dump_label_colors_landcover(os.path.join(output_path, city_folder))
    #     generate_data_info_file(os.path.join(output_path, city_folder))

    # select_from_oem_landcover('/home/yuan/data/syntra/remote_sensing', num_per_city=50)
    # split selected data into train/val/test=4:2:4
    # selected_output_path = '/home/yuan/data/syntra/remote_sensing/oem.landcover_selection'
    # split_selection(selected_output_path)

    ids = list(oem_landcover_cls_id_to_syntra_id.values())
    colors = np.zeros((len(ids), 3), dtype=np.uint8)
    for i, cls_id in enumerate(ids):
        colors[i] = semantic_colors[[k for k, v in semantic_ids.items() if v == cls_id][0]]
    
    print(colors)
