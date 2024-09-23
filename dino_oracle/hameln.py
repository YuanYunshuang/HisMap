import glob
import os
import random

import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset


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


