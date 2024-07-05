import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class HamelnDataset(Dataset):
    def __init__(self, cfgs, mode='train'):
        self.hismap = cv2.imread(cfgs['data_path'])[..., 0] / 255
        self.H, self.W = self.hismap.shape
        self.size = cfgs['img_size']
        samples = np.stack(np.where(self.hismap), axis=-1)
        samples = np.unique(samples // 16, axis=0) * 16
        valid = (samples[:, 0] < self.H - self.size) & (samples[:, 1] < self.W - self.size)
        self.samples = samples[valid]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        x, y = self.samples[item]
        # x = min(max(0, sample[0] + random.randint(-self.size, self.size)), self.H - self.size)
        # y = min(max(0, sample[1] + random.randint(-self.size, self.size)), self.W - self.size)
        sample = self.hismap[x:x+self.size, y:y+self.size]
        return sample

    def collate_batch(self, data):
        img = torch.from_numpy(np.stack(data, axis=0)).unsqueeze(1).float()
        return {'img': img}
