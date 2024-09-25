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
        self.samples = [self.sample_name(x) for x in os.listdir(os.path.join(root, "sample"))]
        self.samples = list(set(self.samples))
        self.root = root
        self.transform = transform
        self.loader = default_loader

    def sample_name(self, img_name):
        patch, year, idx = img_name.split('.')[0].split('_', 2)
        return (patch, idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        patch, idx = self.samples[item]
        files = glob.glob(os.path.join(self.root, 'sample', f'{patch}_*_{idx}.png'))
        sample = [self.loader(f) for f in sorted(files)]

        if len(sample) > 16:
            sample = sample[len(sample) - 16:]
        elif len(sample) < 16:
            sample = sample + sample[len(sample) - 1:] * (16 - len(sample))
        assert len(sample) == 16

        gt = self.loader(os.path.join(self.root, 'gt', f'{patch}_{idx}.png'))
        images = [gt] + sample
        images = self.transform(images)

        return images[1:], images[0]

    def collate_batch(self, data_list):
        return torch.stack(data_list, dim=0)


if __name__ == '__main__':
    dataset = HamelnTemporalDataset("/media/yuan/luna/data/HisMAP/Hameln/train_data", None)
    for sample in dataset:
        pass