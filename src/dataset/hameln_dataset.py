import os
import random

import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset.pipeline import Pipeline


class HamelnLineDataset(Dataset):
    def __init__(self, cfgs, mode='train'):
        self.hismap = cv2.imread(cfgs['data_path'])[..., 0] / 255
        self.H, self.W = self.hismap.shape
        self.size = cfgs['img_size']
        samples = np.stack(np.where(self.hismap), axis=-1)
        samples = np.unique(samples // 4, axis=0) * 4
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


class HamelnDataset(Dataset):
    def __init__(self, cfgs, mode='train'):
        self.root_dir = cfgs['data_path']
        self.gt_dir = cfgs.get('gt_dir', 'gt')
        self.mode = mode
        self.samples = os.listdir(os.path.join(self.root_dir, 'sample'))

        if f'{mode}_patch' in cfgs:
            self.samples = [x for x in self.samples if x.split('_')[0] in cfgs[f'{mode}_patch']]
        if f'{mode}_year' in cfgs:
            self.samples = [x for x in self.samples if x.split('_')[1] in cfgs[f'{mode}_year']]

        print("Loaded {} samples".format(len(self.samples)))

        self.pipeline = Pipeline(cfgs[f'{mode}_pipeline'])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = Image.open(os.path.join(self.root_dir, 'sample', self.samples[item]))
        gt_name = self.samples[item].split('_')
        gt_name.pop(1)
        gt_name = '_'.join(gt_name)
        filename = os.path.join( self.gt_dir, gt_name)
        assert os.path.exists(filename), '{} does not exist'.format(filename)
        gt = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        gt = Image.fromarray(gt)
        data = {
            'filename': self.samples[item],
            'input_img': sample,
            'gt_img': gt,
        }
        data = self.pipeline(data)
        return data

    def collate_batch(self, data_list):
        data_dict = {}
        for k in data_list[0].keys():
            data = [x[k] for x in data_list]
            if isinstance(data[0], torch.Tensor):
                data_dict[k] = torch.stack([x[k] for x in data_list], dim=0)
            else:
                data_dict[k] = data
        return data_dict


