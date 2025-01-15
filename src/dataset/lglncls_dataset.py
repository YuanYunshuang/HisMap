import copy
import os
from fileinput import filename
from random import random
import argparse
import tqdm

import cv2
from PIL import Image
import numpy as np
import torch
from sympy.physics.units import years
from torch.utils.data import DataLoader, Dataset

from src.dataset.pipeline import Pipeline
from src.tools.map_analyzer import read_tiff

CLASSES = ['Woodland', 'Grassland', 'Settlement', 'FlowingWater', 'StandingWater']


def generate_dataset(path_in, path_out, mode='train', crop_size=384, num_pixel_present=100, margin=128):
    # Train data
    path_maps = os.path.join(path_in, mode, 'maps')
    path_labels = os.path.join(path_in, mode, 'label')
    path_out_ = os.path.join(path_out, mode)
    os.makedirs(path_out_, exist_ok=True)

    label_files = os.listdir(path_labels)
    for f in label_files:
        patch, year = f.split('.')[0].split('_')
        if mode == 'test' and year != '1974':
            continue
        label = cv2.imread(os.path.join(path_labels, f), cv2.IMREAD_GRAYSCALE) / 50
        hismap = read_tiff(os.path.join(path_maps, f.replace('png', 'tif')))
        # h, w = max(label.shape[0], hismap.shape[0]), max(label.shape[1], hismap.shape[1])
        h, w = hismap.shape[0], label.shape[1]
        x = np.arange(0, h, margin)
        y = np.arange(0, w, margin)
        xs, ys = np.meshgrid(x, y)
        xs, ys = xs.flatten(), ys.flatten()
        for x1, y1 in zip(xs, ys):
            x2, y2 = x1 + crop_size, y1 + crop_size
            x2 = h if x2 > h else x2
            y2 = w if y2 > w else y2
            L = label[y1:y2, x1:x2]
            M = hismap[y1:y2, x1:x2]

            l_cls = ''
            for i in range(1, 6):
                if np.sum(L == i) > num_pixel_present:
                    l_cls += '1'
                else:
                    l_cls += '0'
            if M.ndim == 2:
                M = M[:, :, None].repeat(1, 1, 3)
            M = np.pad(M, ((0, crop_size - M.shape[0]), (0, crop_size - M.shape[1]), (0, 0)), mode='constant', constant_values=0)
            cv2.imwrite(os.path.join(path_out_, f"{patch}_{year}_{x1}_{y1}_{l_cls}.jpg"), M)


class LglnclsDataset(Dataset):

    def __init__(self,
                 cfgs,
                 mode='train',
    ):
        self.cfgs = cfgs
        self.root_dir = cfgs['data_path']
        self.mode = mode

        self.img_dir = os.path.join(self.root_dir, self.mode)
        self.samples = os.listdir(self.img_dir)

        self.pipeline = Pipeline(cfgs[f'{mode}_pipeline'])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        filename = self.samples[item]
        patch, year, x , y, l = filename.split('.')[0].split('_')
        img = cv2.imread(os.path.join(self.img_dir, filename), cv2.IMREAD_UNCHANGED)

        data = {
            'patch': patch,
            'year': year,
            'x': x,
            'y': y,
            'img': img,
            'label': int(l[0]),
        }
        data = self.pipeline(data)
        return data

    def collate_batch(self, data_list):
        data_dict = {}
        for k in data_list[0].keys():
            data = [x[k] for x in data_list]
            if isinstance(data[0], torch.Tensor):
                data_dict[k] = torch.stack(data, dim=0)
            elif isinstance(data[0], list) and isinstance(data[0][0], torch.Tensor):
                data = [torch.stack(x, dim=0) for x in data]
                data_dict[k] = torch.cat(data, dim=0)
            elif k == 'label':
                data_dict[k] = torch.tensor(data, dtype=torch.long)
            else:
                data_dict[k] = data

        return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-in", type=str)
    parser.add_argument("--path-out", type=str)
    parser.add_argument("--crop-size", type=int, default=384)
    args = parser.parse_args()

    for mode in ['train', 'test']:
        generate_dataset(
            args.path_in,
            args.path_out,
            mode=mode,
            crop_size=384,
        )

    # from src.utils.misc import load_yaml
    # cfg = load_yaml('../config/cvit/demo.yaml')
    # data_cfg = cfg['DATASET']
    # dataset = LglnclsDataset(data_cfg, mode='train')
    # dataloader = DataLoader(dataset, batch_size=2, num_workers=4, collate_fn=dataset.collate_batch)
    # for d in dataloader:
    #     print(d)