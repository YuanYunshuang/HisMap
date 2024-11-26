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
                 remapped_label=True,
                 load_to_memory=False
    ):
        self.root_dir = cfgs['data_path']
        self.map_dir = os.path.join(self.root_dir, mode, 'maps')
        self.img_dir = os.path.join(self.root_dir, mode, 'imgs')
        self.gt_dir = os.path.join(self.root_dir, mode, 'gt')
        self.label_dir = os.path.join(self.root_dir, mode, 'label')
        self.mode = mode
        self.crop_step_size = cfgs['crop_step_size']
        self.crop_size = cfgs['crop_size']
        self.max_num_img = cfgs.get('max_num_img', 4)
        self.load_to_memory = load_to_memory
        if 'years' in cfgs:
            self.years = [int(year) for year in cfgs['years'].split(',')]
        else:
            start_year = cfgs['start_year']
            end_year = cfgs['end_year']
            self.years = [y for y in range(start_year, end_year+1)]
        if load_to_memory:
            self.load_maps()
        else:
            self.parse_maps()
        self.load_labels()
        self.pipeline = Pipeline(cfgs[f'{mode}_pipeline'])

    def load_maps(self):
        # parse maps in valid age
        maps = {}
        self.samples = []
        print("Loading maps ...")
        for x in tqdm.tqdm(sorted(os.listdir(self.map_dir))):
            patch, year = x.split('.')[0].split('_')
            patch, year = int(patch), int(year)
            if patch not in maps:
                maps[patch] = {'years': [], 'size': [], 'data': []}
            if year in self.years:
                maps[patch]['years'].append(year)
                TIF = read_tiff(os.path.join(self.map_dir, x))
                maps[patch]['size'].append(TIF.shape)
                maps[patch]['data'].append(TIF)
                map_size = TIF.shape
                x = np.arange(0, map_size[0] - self.crop_size + 1, self.crop_step_size)
                y = np.arange(0, map_size[1] - self.crop_size + 1, self.crop_step_size)
                xy = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
                self.samples.extend(np.concatenate([np.zeros_like(xy[:, :1]) + patch,
                                                   np.zeros_like(xy[:, :1]) + year, xy], axis=-1))
        print(f"Cropped the maps into {len(self.samples)} samples.")

    def parse_maps(self):
        # parse maps in valid age
        maps = {}
        print("Parsing maps to cropped images...")
        self.samples = []
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
            for filename in tqdm.tqdm(sorted(os.listdir(self.map_dir))):
                patch, year = filename.split('.')[0].split('_')
                patch, year = int(patch), int(year)
                TIF = read_tiff(os.path.join(self.map_dir, filename))
                map_size = TIF.shape[:2]
                x = np.arange(0, map_size[0] - self.crop_size + 1, self.crop_step_size)
                y = np.arange(0, map_size[1] - self.crop_size + 1, self.crop_step_size)
                xys = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
                for xy in xys:
                    img = TIF[xy[0]:xy[0]+self.crop_size, xy[1]:xy[1]+self.crop_size]
                    cv2.imwrite(os.path.join(self.img_dir, f'{patch}_{year}_{xy[0]}_{xy[1]}.jpg'), img)
                if year in self.years:
                    self.samples.extend(np.concatenate([np.zeros_like(xy[:, :1]) + patch,
                                                        np.zeros_like(xy[:, :1]) + year, xy], axis=-1))
        else:
            for x in os.listdir(self.img_dir):
                patch, year, x, y = x.split('.')[0].split('_')
                if int(year) in self.years:
                    self.samples.append([int(patch), int(year), int(x), int(y)])

        print(f"Found {len(self.samples)} image samples.")

    def load_gts(self):
        # parse labels
        labels = {}
        print("\nLoading labels ...")
        for x in tqdm.tqdm(sorted([f for f in os.listdir(self.gt_dir) if self.CLASSES[0] in f])):
            patch, year, _ = x.split('.')[0].split('_')
            patch, year = int(patch), int(year)
            if patch not in labels:
                labels[patch] = {'years': [], 'size': [], 'data': []}
            if year in self.years:
                TIFs = []
                for cls in self.CLASSES:
                    filename = x.replace(self.CLASSES[0], cls)
                    TIF = read_tiff(os.path.join(self.gt_dir, filename))
                    if TIF is not None:
                        TIFs.append(TIF==1)
                    else:
                        # when file not exists, assume all labels are background
                        assert len(TIFs) > 0
                        TIFs.append(np.zeros(TIFs[-1].shape, dtype=bool))

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

    def load_labels(self):
        labels = {}
        print("\nLoading labels ...")
        for x in tqdm.tqdm(sorted(os.listdir(self.label_dir))):
            patch, year = x.split('.')[0].split('_')
            patch, year = int(patch), int(year)
            if patch not in labels:
                labels[patch] = {'years': [], 'size': [], 'data': []}
            if year in self.years:
                cur_label = cv2.imread(os.path.join(self.label_dir, x), cv2.IMREAD_UNCHANGED) / 50
                labels[patch]['data'].append(cur_label)
                labels[patch]['years'].append(year)
                labels[patch]['size'].append(cur_label.shape)
        self.labels = labels


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        patch, year, x, y = self.samples[item]
        filename = f"{patch}_{year}_{x}_{y}.jpg"
        img = cv2.imread(os.path.join(self.img_dir, filename), cv2.IMREAD_COLOR)
        img = Image.fromarray(img)
        gt = []
        for label_year, label in zip(self.labels[patch]['years'], self.labels[patch]['data']):
            cur_label = label[x:x+self.crop_size, y:y+self.crop_size]
            cur_label = np.pad(cur_label,
                               ((0, max(self.crop_size - cur_label.shape[0], 0)),
                                (0, max(self.crop_size - cur_label.shape[1], 0)),
                                ), mode='constant', constant_values=0)
            gt.append(Image.fromarray(cur_label))
        data = {
            'patch': patch,
            'data_years': [year],
            'x': x,
            'y': y,
            'gt_years': self.labels[patch]['years'],
            'input_img': [img],
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


class LglnSeqDataset(Dataset):
    CLASSES = ['FlowingWater', 'Grassland', 'Settlement', 'StandingWater', 'Woodland']
    def __init__(self,
                 cfgs,
                 mode='train',
                 remapped_label=True,
                 load_maps_to_memory=False
    ):
        self.root_dir = cfgs['data_path']
        self.map_dir = os.path.join(self.root_dir, mode, 'maps')
        self.img_dir = os.path.join(self.root_dir, mode, 'imgs')
        self.gt_dir = os.path.join(self.root_dir, mode, 'gt')
        self.label_dir = os.path.join(self.root_dir, mode, 'label')
        self.mode = mode
        self.crop_step_size = cfgs['crop_step_size']
        self.crop_size = cfgs['crop_size']
        self.max_num_img = cfgs.get('max_num_img', 4)
        self.load_maps_to_memory = load_maps_to_memory
        if 'years' in cfgs:
            self.years = [int(year) for year in cfgs['years'].split(',')]
        else:
            start_year = cfgs['start_year']
            end_year = cfgs['end_year']
            self.years = [y for y in range(start_year, end_year+1)]
        if load_maps_to_memory:
            self.load_maps()
        else:
            self.parse_maps()
        if remapped_label:
            self.load_labels()
        else:
            self.load_gts()

        self.pipeline = Pipeline(cfgs[f'{mode}_pipeline'])

    def load_maps(self):
        # parse maps in valid age
        maps = {}
        print("Loading maps ...")
        cnt = 0
        for x in tqdm.tqdm(sorted(os.listdir(self.map_dir))):
            patch, year = x.split('.')[0].split('_')
            patch, year = int(patch), int(year)
            if patch not in maps:
                maps[patch] = {'years': [], 'size': [], 'data': []}
            if year in self.years:
                cnt += 1
                maps[patch]['years'].append(year)
                TIF = read_tiff(os.path.join(self.map_dir, x))
                maps[patch]['size'].append(TIF.shape)
                maps[patch]['data'].append(TIF)
        self.maps = maps
        print("Loaded {} maps.".format(cnt))

        self.samples = []
        for patch, info in maps.items():
            map_size = info['size'][0][:2]
            x = np.arange(0, map_size[0] - self.crop_size + 1, self.crop_step_size)
            y = np.arange(0, map_size[1] - self.crop_size + 1, self.crop_step_size)
            xy = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
            self.samples.extend(np.concatenate([np.zeros_like(xy[:, :1]) + patch, xy], axis=-1))
        print(f"Cropped the maps into {len(self.samples)} samples.")

    def parse_maps(self):
        # parse maps in valid age
        patches = {}
        print("Parsing maps to cropped images...")
        self.samples = []
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
            for filename in tqdm.tqdm(sorted(os.listdir(self.map_dir))):
                patch, year = filename.split('.')[0].split('_')
                patch, year = int(patch), int(year)
                TIF = read_tiff(os.path.join(self.map_dir, filename))
                map_size = TIF.shape[:2]
                x = np.arange(0, map_size[0] - self.crop_size + 1, self.crop_step_size)
                y = np.arange(0, map_size[1] - self.crop_size + 1, self.crop_step_size)
                xys = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
                for xy in xys:
                    img = TIF[xy[0]:xy[0]+self.crop_size, xy[1]:xy[1]+self.crop_size]
                    cv2.imwrite(os.path.join(self.img_dir, f'{patch}_{year}_{xy[0]}_{xy[1]}.jpg'), img)
                if patch not in patches:
                    self.samples.extend(np.concatenate([np.zeros_like(xy[:, :1]) + patch, xy], axis=-1))
        print(f"Found {len(self.samples)} image samples.")

    def load_gts(self):
        # parse labels
        labels = {}
        print("\nLoading labels ...")
        for x in tqdm.tqdm(sorted([f for f in os.listdir(self.gt_dir) if self.CLASSES[0] in f])):
            patch, year, _ = x.split('.')[0].split('_')
            patch, year = int(patch), int(year)
            if patch not in labels:
                labels[patch] = {'years': [], 'size': [], 'data': []}
            if year in self.years:
                TIFs = []
                for cls in self.CLASSES:
                    filename = x.replace(self.CLASSES[0], cls)
                    TIF = read_tiff(os.path.join(self.gt_dir, filename))
                    if TIF is not None:
                        TIFs.append(TIF==1)
                    else:
                        # when file not exists, assume all labels are background
                        assert len(TIFs) > 0
                        TIFs.append(np.zeros(TIFs[-1].shape, dtype=bool))

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

    def load_labels(self):
        labels = {}
        print("\nLoading labels ...")
        for x in tqdm.tqdm(sorted(os.listdir(self.label_dir))):
            patch, year = x.split('.')[0].split('_')
            patch, year = int(patch), int(year)
            if patch not in labels:
                labels[patch] = {'years': [], 'size': [], 'data': []}
            if year in self.years:
                cur_label = cv2.imread(os.path.join(self.label_dir, x), cv2.IMREAD_UNCHANGED)
                map_size = self.maps[patch]['size'][0]
                cur_label = np.pad(cur_label,
                                   ((0, max(map_size[0] - cur_label.shape[0], 0)),
                                    (0, max(map_size[1] - cur_label.shape[1], 0)),
                                    ), mode='constant', constant_values=0)
                labels[patch]['data'].append(cur_label)
                labels[patch]['years'].append(year)
                labels[patch]['size'].append(cur_label.shape)
        self.labels = labels


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        patch, x, y = self.samples[item]
        years = self.maps[patch]['years']
        sample_inds = np.random.choice(np.arange(len(years)), min(self.max_num_img, len(years)), replace=False)
        years = [years[ind] for ind in sample_inds]
        data_list = [self.maps[patch]['data'][ind] for ind in sample_inds]
        sample = []
        gt = []
        # gt_inds = []
        for data in data_list:
            img = data[x:x+self.crop_size, y:y+self.crop_size]
            if img.ndim == 2:
                img = img[:, :, None].repeat(3, axis=2)
            sample.append(Image.fromarray(img))
        for label_year, label in zip(self.labels[patch]['years'], self.labels[patch]['data']):
            gt.append(Image.fromarray(label[x:x+self.crop_size, y:y+self.crop_size]))
            # gt_inds.append(years.index(label_year))
        data = {
            'patch': patch,
            'data_years': years,
            'x': x,
            'y': y,
            'gt_years': self.labels[patch]['years'],
            # 'gt_inds': gt_inds,
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