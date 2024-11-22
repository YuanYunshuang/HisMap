import os
from fileinput import filename

import tqdm

import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset.pipeline import Pipeline
from src.tools.map_analyzer import read_tiff


class LglnDataset(Dataset):
    CLASSES = ['FlowingWater', 'Grassland', 'Settlement', 'StandingWater', 'Woodland']
    def __init__(self,
                 cfgs,
                 mode='train',
    ):
        self.root_dir = cfgs['data_path']
        self.map_dir = os.path.join(self.root_dir, mode, 'maps')
        self.gt_dir = os.path.join(self.root_dir, mode, 'gt')
        self.mode = mode
        self.crop_step_size = cfgs['crop_step_size']
        self.crop_size = cfgs['crop_size']
        self.start_year = cfgs['start_year']
        self.end_year = cfgs['end_year']

        # parse maps in valid age
        maps = {}
        print("Loading maps...")
        for x in tqdm.tqdm(sorted(os.listdir(self.map_dir))):
            patch, year = x.split('.')[0].split('_')
            patch, year = int(patch), int(year)
            if patch not in maps:
                maps[patch] = {'years': [], 'size': [], 'data': []}
            if year >= self.start_year and year <= self.end_year:
                maps[patch]['years'].append(year)
                TIF = read_tiff(os.path.join(self.map_dir, x))
                maps[patch]['size'].append(TIF.shape)
                maps[patch]['data'].append(TIF)
        self.maps = maps

        self.samples = []
        for patch, info in maps.items():
            map_size = info['size'][0][:2]
            x = np.arange(0, map_size[0] - self.crop_size + 1, self.crop_step_size)
            y = np.arange(0, map_size[1] - self.crop_size + 1, self.crop_step_size)
            xy = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
            self.samples.extend(np.concatenate([np.zeros_like(xy[:, :1]) + patch, xy], axis=-1))

        # parse labels
        labels = {}
        print("\nLoading labels...")
        for x in tqdm.tqdm(sorted([f for f in os.listdir(self.gt_dir) if self.CLASSES[0] in f])):
            patch, year, _ = x.split('.')[0].split('_')
            patch, year = int(patch), int(year)
            if patch not in labels:
                labels[patch] = {'years': [], 'size': [], 'data': []}
            if year >= self.start_year and year <= self.end_year:
                TIFs = []
                for cls in self.CLASSES:
                    filename = x.replace(self.CLASSES[0], cls)
                    TIF = read_tiff(os.path.join(self.gt_dir, filename))
                    TIFs.append(TIF==1)
                TIFs = np.stack(TIFs, axis=-1)
                cur_label = np.packbits(TIFs, axis=-1, bitorder='little').squeeze(-1)
                map_size = self.maps[patch]['size'][0]
                cur_label = np.pad(cur_label,
                                   ((0, max(map_size[0] - cur_label.shape[0], 0)),
                                    (0, max(map_size[1] - cur_label.shape[1], 0)),
                                    ), mode='constant', constant_values=0)
                labels[patch]['data'].append(cur_label)
                labels[patch]['years'].append(year)
                labels[patch]['size'].append(TIFs.shape)
        self.labels = labels

        self.pipeline = Pipeline(cfgs[f'{mode}_pipeline'])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        patch, x, y = self.samples[item]
        years = self.maps[patch]['years']
        sample = []
        gt = []
        for data in self.maps[patch]['data']:
            sample.append(Image.fromarray(data[x:x+self.crop_size, y:y+self.crop_size]))
        for label in self.labels[patch]['data']:
            gt.append(Image.fromarray(label[x:x+self.crop_size, y:y+self.crop_size]))
        data = {
            'patch': patch,
            'data_years': years,
            'gt_years': self.labels[patch]['years'],
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
                data_dict[k] = torch.stack(data, dim=0)
            elif isinstance(data[0], list) and isinstance(data[0][0], torch.Tensor):
                data = [torch.stack(x, dim=0) for x in data]
                data_dict[k] = torch.cat(data, dim=0)
            else:
                data_dict[k] = data
        return data_dict


if __name__ == '__main__':
    dataset = LglnDataset(data_path="/home/yuan/data/HisMap/LGLN/dataset")
    for d in dataset:
        print(d)