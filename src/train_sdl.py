import random

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.sdl.splinenet import SplineNet
from src.losses.l1_loss import SmoothL1Loss


class HisMAPDataset(Dataset):
    def __init__(self, map_path, size=128):
        self.hismap = cv2.imread(map_path)[..., 0] / 255
        self.H, self.W = self.hismap.shape
        self.size = size
        samples = np.stack(np.where(self.hismap), axis=-1)
        valid = (samples[:, 0] < self.H - self.size) & (samples[:, 1] < self.W - self.size)
        self.samples = samples[valid]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        x = min(max(0, sample[0] + random.randint(-self.size, self.size)), self.H - self.size)
        y = min(max(0, sample[1] + random.randint(-self.size, self.size)), self.W - self.size)
        sample = self.hismap[x:x+self.size, y:y+self.size]
        return sample


####PARAMETERS####
batch_size = 64
map_path = "/home/data/HisMAP/full_map/3922_2017.tif"
epoch = 10


####MODULES####
dataloader = DataLoader(
    dataset=Dataset(map_path),
    batch_size=batch_size,
    num_workers=4,
    shuffle=True,
    pin_memory=True
)

net = SplineNet().cuda()
net.train()
criterion = SmoothL1Loss()


