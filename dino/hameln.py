import glob
import os
import random

import cv2
import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from src.tools.map_analyzer import read_tiff


class HamelnDataset(Dataset):
    def __init__(self, root, transform):
        self.samples = glob.glob(os.path.join(root, "*.png"))
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        path = self.samples[item]
        sample = self.loader(path)
        sample = self.transform(sample)
        return sample, 0

    def collate_batch(self, data_list):
        data_dict = {}
        for k in data_list[0].keys():
            data = [x[k] for x in data_list]
            if isinstance(data[0], torch.Tensor):
                data_dict[k] = torch.stack([x[k] for x in data_list], dim=0)
            else:
                data_dict[k] = data
        return data_dict


class HamelnTemporalDataset(Dataset):
    def __init__(self, root, transform):
        self.parse_maps(root)
        self.samples = glob.glob(os.path.join(root, "*.png"))
        self.transform = transform
        self.loader = default_loader

    def parse_maps(self, root):
        files = glob.glob(os.path.join(root, "*.tif"))
        maps = {'3922': [], '3921': [], '3822': [], '3821': []}
        years = {'3922': [], '3921': [], '3822': [], '3821': []}
        for f in tqdm.tqdm(files):
            try:
                patch, year = os.path.basename(f).split('.')[0].split('_')
            except:
                continue
            m = read_tiff(f)
            maps[patch].append(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY))
            years[patch].append(year)

        for k, v in maps.items():
            cur_years = years[k]
            sort_inds = np.argsort(cur_years)
            years[k] = cur_years[sort_inds]
            maps[k] = v[sort_inds]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        path = self.samples[item]
        sample = self.loader(path)
        sample = self.transform(sample)
        return sample, 0

    def collate_batch(self, data_list):
        data_dict = {}
        for k in data_list[0].keys():
            data = [x[k] for x in data_list]
            if isinstance(data[0], torch.Tensor):
                data_dict[k] = torch.stack([x[k] for x in data_list], dim=0)
            else:
                data_dict[k] = data
        return data_dict


if __name__ == '__main__':
    dataset = HamelnTemporalDataset("/media/yuan/luna/data/HisMAP/Hameln/TK_projektiv_entzerrt_1m", None)
    for sample in dataset:
        pass