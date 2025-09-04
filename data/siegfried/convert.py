import os
import cv2
import numpy as np
import tqdm
import json
from geotiff import GeoTiff
from scipy.ndimage import zoom


def read_tiff(filename):
    if not os.path.exists(filename):
        print('File not found: {}'.format(filename))
        return None
    geo_tiff = GeoTiff(filename)
    tiff_array = np.array(geo_tiff.read())
    return tiff_array


def convert_siegfried_to_smol(siegfried_folder, smoldata_folder):
    dinfo = {'train': {}, 'val': {}, 'test': {}}
    split_list = {'train': [], 'val': [], 'test': []}
    os.makedirs(os.path.join(smoldata_folder, 'siegfried', 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(smoldata_folder, 'siegfried', 'lbls'), exist_ok=True)
    # zoom_ratio = 384 / 224
    for data_name in os.listdir(siegfried_folder):
        data_name_prefix = data_name.replace('dataset_', '')
        cur_label = '1' if data_name_prefix == 'railway' else '2'
        data_path = os.path.join(siegfried_folder, data_name)
        # convert images
        for split in ['train', 'val', 'test']:
            print('>>>>>>>>>>>>>>>>>>SPLIT {} {}'.format(data_name_prefix, split))
            for img_name in tqdm.tqdm(os.listdir(os.path.join(data_path, split))):
                img_path = os.path.join(data_path, split, img_name)
                img = cv2.imread(img_path)
                # img = zoom(img, (zoom_ratio, zoom_ratio, 1), order=3)
                cv2.imwrite(os.path.join(smoldata_folder, 'siegfried', 'imgs', data_name_prefix + '_' + img_name.replace('.tif', '.png')), img)
                lbl_path = os.path.join(data_path, 'annotation', split, img_name)
                lbl = cv2.imread(lbl_path,cv2.IMREAD_GRAYSCALE)
                # lbl = zoom(lbl, (zoom_ratio, zoom_ratio), order=0)
                lbl[lbl>0] = int(cur_label)
                cv2.imwrite(os.path.join(smoldata_folder, 'siegfried', 'lbls', data_name_prefix + '_'  + img_name.replace('.tif', '.png')), lbl)

                dinfo[split][f'siegfried.{data_name_prefix}_' + img_name.replace('.tif', '')] = {
                    '0': np.sum(lbl==0) / (lbl.shape[0] * lbl.shape[1]),
                    '1': 0,
                    '2': 0,
                    '3': 0,
                    '4': 0,
                }
                dinfo[split][f'siegfried.{data_name_prefix}_' + img_name.replace('.tif', '')][cur_label] = np.sum(lbl == int(cur_label)) / (lbl.shape[0] * lbl.shape[1])

                split_list[split].append(f'siegfried.{data_name_prefix}_' + img_name.replace('.tif', ''))

    for k, v in dinfo.items():
        with open(os.path.join(smoldata_folder, f'dinfo_siegfried_{k}.json'), 'w') as f:
            json.dump(v, f, indent=4)
        with open(os.path.join(smoldata_folder, f'{k}_siegfried.txt'), 'w') as f:
            f.write('\n'.join(split_list[k]))
        with open(os.path.join(smoldata_folder, f'{k}_siegfried_railway.txt'), 'w') as f:
            f.write('\n'.join([x for x in split_list[k] if 'railway' in x]))
        with open(os.path.join(smoldata_folder, f'{k}_siegfried_vineyard.txt'), 'w') as f:
            f.write('\n'.join([x for x in split_list[k] if 'vineyard' in x]))


def generate_partial_list(siegfried_folder, smoldata_folder):
    for data_name in ['railway', 'vineyard']:
        for amount in ['partial_10', 'partial_1', 'fewshot10']:
            for split in ['train', 'val']:
                path = os.path.join(siegfried_folder, f'dataset_{data_name}', amount, split)
                if not os.path.exists(path):
                    continue
                file_list = os.listdir(path)
                with open(os.path.join(smoldata_folder, f'{split}_siegfried_{data_name}_{amount}.txt'), 'w') as f:
                    f.write('\n'.join([f'siegfried.{data_name}_' + x[:-4] for x in file_list]))


if __name__ == '__main__':
    # convert_siegfried_to_smol(
    #     "/home/yuan/data/HisMap/Siegfried_Railway_Vineyard",
    #     "/home/yuan/data/HisMap/SMOLdata"
    # )
    generate_partial_list(
        "/home/yuan/data/HisMap/Siegfried_Railway_Vineyard",
        "/home/yuan/data/HisMap/SMOLdata"
    )