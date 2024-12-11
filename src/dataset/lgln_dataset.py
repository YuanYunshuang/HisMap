import copy
import os
from fileinput import filename

import tqdm

import cv2
from PIL import Image
import numpy as np
import torch
from sympy.physics.units import years
from torch.utils.data import DataLoader, Dataset

from src.dataset.pipeline import Pipeline
from src.tools.map_analyzer import read_tiff


class LglnDataset(Dataset):
    CLASSES = ['Woodland', 'Grassland', 'Settlement', 'FlowingWater', 'StandingWater']
    def __init__(self,
                 cfgs,
                 mode='train',
    ):
        self.cfgs = cfgs
        self.root_dir = cfgs['data_path']
        self.mode = mode
        self.crop_step_size = cfgs['crop_step_size']
        self.crop_size = cfgs['crop_size']
        self.max_num_img = cfgs.get('max_num_img', 4)
        self.labels_info = cfgs.get('labels', None)
        self.load_to_memory = cfgs.get('load_to_memory', False)
        # finetune: gradually fine-tune with pseudo labels by age-tracing
        # all: train on all images across all ages at once, using label only from one year
        # pre: pre-train on the images that have ground truth labels
        self.train_mode = cfgs.get('train_mode', 'pre')
        self.reparse_pseudo_label = cfgs.get('reparse_pseudo_label', False)

        self.init_paths()
        self.parse_meta(cfgs)
        if self.load_to_memory or self.mode == 'test':
            self.load_maps()
            self.load_labels()
        else:
            self.parse_maps()
            self.parse_labels()

        if self.labels_info is None:
            self.update_label_info()

        self.train_pipeline = Pipeline(cfgs['train_pipeline'])
        self.test_pipeline = Pipeline(cfgs['test_pipeline'])

    def init_paths(self):
        self.map_dir = os.path.join(self.root_dir, self.mode, 'maps')
        self.img_dir = os.path.join(self.root_dir, self.mode, 'imgs')
        if self.cfgs['start_year'] == 2023 and self.mode == 'train':
            self.gt_dir = os.path.join(self.root_dir, self.mode, 'gt2023')
            self.label_dir = os.path.join(self.root_dir, self.mode, 'label2023')
            dirname = 'pseudo_label2023'
        else:
            self.gt_dir = os.path.join(self.root_dir, self.mode, 'gt')
            self.label_dir = os.path.join(self.root_dir, self.mode, 'label')
            dirname = 'pseudo_label'
        self.pseudo_label_dir = os.path.join(self.root_dir, self.mode, dirname)
        os.makedirs(self.pseudo_label_dir, exist_ok=True)

    def parse_meta(self, cfgs):
        self.meta = {}
        years = []
        for x in sorted(os.listdir(self.map_dir)):
            patch, year = x.split('.')[0].split('_')
            patch, year = int(patch), int(year)
            if self.train_mode == 'trace' and year == 2023:
                continue
            if patch not in self.meta:
                self.meta[patch] = []
            years.append(year)
            self.meta[patch].append(year)
        for k, v in self.meta.items():
            self.meta[k] = sorted(v)
        if 'years' in cfgs:
            years = cfgs['years'].split(',') if isinstance(cfgs['years'], str) else cfgs['years']
            self.years = [int(year) for year in years]
        else:
            start_year = cfgs['start_year']
            end_year = cfgs['end_year']
            years = list(set(years))
            self.years = [y for y in years if (y >= start_year) and (y <= end_year)]

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
                x = np.arange(0, map_size[0], self.crop_step_size)
                y = np.arange(0, map_size[1], self.crop_step_size)
                xy = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
                self.samples.extend(np.concatenate([np.zeros_like(xy[:, :1]) + patch,
                                                   np.zeros_like(xy[:, :1]) + year, xy], axis=-1))
        self.maps = maps
        print(f"Cropped the maps into {len(self.samples)} samples.")

    def parse_maps(self):
        # parse maps in valid age
        self.maps = {}
        print("Parsing maps images...")
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
            samples = []
            for x in os.listdir(self.img_dir):
                patch, year, x, y = x.split('.')[0].split('_')
                patch, year, x, y = int(patch), int(year), int(x), int(y)
                samples.append([patch, year, x, y])
            if self.train_mode == 'trace':
                self.traced_ages = self.trace_ages(mix_old=True)
                for s in samples:
                    if s[1] in self.traced_ages[s[0]]['years']:
                        self.samples.append(s)
            elif self.train_mode == "all":
                self.samples = samples
            elif self.train_mode == "pre":
                self.samples = [s for s in samples if s[1] in self.years]
            else:
                raise NotImplementedError
            # self.samples = self.samples[:16*20]
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

    def load_labels(self, meta_only=False):
        labels = {}
        print("\nLoading labels ...")
        for x in tqdm.tqdm(sorted(os.listdir(self.label_dir))):
            patch, year = x.split('.')[0].split('_')
            patch, year = int(patch), int(year)

            if patch not in labels:
                if meta_only:
                    labels[patch] = {'years': [], 'size': []}
                else:
                    labels[patch] = {'years': [], 'size': [], 'data': []}
            if year in self.years:
                cur_label = cv2.imread(os.path.join(self.label_dir, x), cv2.IMREAD_UNCHANGED) / 50
                if not meta_only:
                    labels[patch]['data'].append(cur_label.astype(np.int8))
                labels[patch]['years'].append(year)
                labels[patch]['size'].append(cur_label.shape)
        self.labels = labels

    def parse_labels(self):
        if self.reparse_pseudo_label:
            self.load_labels()
            for info in self.samples:
                patch, year, x, y = info
                if year not in self.labels[patch]['years']:
                    continue
                idx = self.labels[patch]['years'].index(year)
                label = self.labels[patch]['data'][idx][x:x + self.crop_size, y:y + self.crop_size]
                label = np.pad(label,
                               ((0, max(self.crop_size - label.shape[0], 0)),
                                (0, max(self.crop_size - label.shape[1], 0)),
                                ), mode='constant', constant_values=0)
                filename = f"{patch}_{year}_{x}_{y}.png"
                cv2.imwrite(os.path.join(self.pseudo_label_dir, filename), label)
            for k, v in self.labels.items():
                v.pop('data')
        else:
            self.load_labels(meta_only=True)
        for k, v in self.meta.items():
            for y in v:
                if y in self.years and y not in self.labels[k]['years']:
                    self.labels[k]['years'].append(y)
            self.labels[k]['years'] = sorted(self.labels[k]['years'])

    def update_label_info(self):
        files = os.listdir(self.pseudo_label_dir)
        self.labels_info = {}
        for f in files:
            patch, year, x, y = f.split('.')[0].split('_')
            patch, year = int(patch), int(year)
            if patch not in self.labels_info:
                self.labels_info[patch] = []
            if year not in self.labels_info[patch]:
                self.labels_info[patch].append(year)

    def cfgs_to_pseudo_label_mode(self):
        self.original_config = {
            'mode': self.mode,
            'crop_size': self.crop_size,
            'crop_step_size': self.crop_step_size,
            'load_to_memory': self.load_to_memory,
        }
        self.mode = 'test'
        self.load_to_memory = True
        self.crop_size = 1024
        self.crop_step_size = 1024

    def trace_ages(self, mix_old=False):
        traced_ages = {}
        for k, v in self.labels_info.items():
            start_year = min(v)
            end_year = max(v)
            start_idx = self.meta[k].index(start_year) - 1
            end_idx = self.meta[k].index(end_year) + 1
            traced_ages[k] = {'years': [], 'label_years': []}

            if start_idx >= 0:
                traced_ages[k]['years'].append(self.meta[k][start_idx])
                traced_ages[k]['label_years'].append(start_year)
            if end_idx < len(self.meta[k]):
                traced_ages[k]['years'].append(self.meta[k][end_idx])
                traced_ages[k]['label_years'].append(end_year)
            if mix_old:
                traced_ages[k]['years'].extend(v)
                traced_ages[k]['label_years'].extend(v)
        return traced_ages

    def pre_update_pseudo_labels(self, step=1):
        self.cfgs_to_pseudo_label_mode()
        pseudo_labels = {}
        has_new_labels = False
        self.maps = {}
        self.samples = []
        traced_ages = self.trace_ages()

        for k, v in traced_ages.items():
            pseudo_labels[k] = {'years': v['years'], 'data': []}
            has_new_labels = True
        # load maps for pseudo labels
        if has_new_labels:
            print("Loading maps for generating pseudo labels ...")
            for k, v in pseudo_labels.items():
                self.maps[k] = {'years': [], 'size': [], 'data': []}
                for year in v['years']:
                    TIF = read_tiff(os.path.join(self.map_dir, f"{k}_{year}.tif"))
                    self.maps[k]['years'].append(year)
                    self.maps[k]['size'].append(TIF.shape)
                    self.maps[k]['data'].append(TIF)
                    v['data'].append(np.zeros(TIF.shape[:2], dtype=np.int8))
                    map_size = TIF.shape
                    x = np.arange(0, map_size[0], self.crop_size)
                    y = np.arange(0, map_size[1], self.crop_size)
                    xy = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
                    self.samples.extend(np.concatenate([np.zeros_like(xy[:, :1]) + k,
                                                        np.zeros_like(xy[:, :1]) + year, xy], axis=-1))
        return pseudo_labels, has_new_labels

    def post_update_pseudo_labels(self, pseudo_labels):
        for k, v in self.original_config.items():
            setattr(self, k, v)
        for k, v in pseudo_labels.items():
            for i, year in enumerate(v['years']):
                map_size = self.maps[k]['size'][i]
                x = np.arange(0, map_size[0], self.crop_step_size)
                y = np.arange(0, map_size[1], self.crop_step_size)
                xys = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
                for xy in xys:
                    label = v['data'][i][xy[0]:xy[0]+self.crop_size, xy[1]:xy[1]+self.crop_size]
                    filename = os.path.join(self.pseudo_label_dir, f"{k}_{year}_{xy[0]}_{xy[1]}.png")
                    cv2.imwrite(filename, label)
                # self.labels[k]['years'].append(year)
                # self.labels[k]['size'].append(self.labels[k]['size'][0])
                # self.labels_info[k].append(year)

    def get_label(self, patch, year, x, y):
        if self.train_mode == 'all':
            year = self.labels_info[patch][0]
        elif self.train_mode == 'trace':
            year = self.traced_ages[patch]['label_years'][self.traced_ages[patch]['years'].index(year)]
        if patch in self.labels and 'data' in self.labels[patch] and year in self.labels[patch]['years']:
            idx = self.labels[patch]['years'].index(year)
            label = self.labels[patch]['data'][idx][x:x + self.crop_size, y:y + self.crop_size]
            label = np.pad(label,
                               ((0, max(self.crop_size - label.shape[0], 0)),
                                (0, max(self.crop_size - label.shape[1], 0)),
                                ), mode='constant', constant_values=0)
        else:
            file = os.path.join(self.pseudo_label_dir, f"{patch}_{year}_{x}_{y}.png")
            if os.path.exists(file):
                label = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            else:
                label = None


        return label

    def get_img(self, patch, year, x, y):
        if hasattr(self, 'maps') and patch in self.maps:
            img = self.maps[patch]['data'][self.maps[patch]['years'].index(year)][
                  x:x+self.crop_size, y:y+self.crop_size]
            if img.ndim == 2:
                img = img[:, :, np.newaxis].repeat(3, axis=2)
        else:
            filename = f"{patch}_{year}_{x}_{y}.jpg"
            img = cv2.imread(os.path.join(self.img_dir, filename), cv2.IMREAD_COLOR)

        img = np.pad(img,
                     ((0, max(self.crop_size - img.shape[0], 0)),
                      (0, max(self.crop_size - img.shape[1], 0)),
                      (0, 0)), mode='constant', constant_values=0)
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        patch, year, x, y = self.samples[item]
        img = self.get_img(patch, year, x, y)
        label = self.get_label(patch, year, x, y)
        imgs = [Image.fromarray(img)]
        gt = []
        if label is not None:
            gt = [Image.fromarray(label)]
        elif self.mode == 'train':
            print(f"No label found: patch {patch}, year {year}, x {x}, y {y}")

        data = {
            'patch': patch,
            'data_years': [year],
            'x': x,
            'y': y,
            'gt_years': [year],
            'input_img': imgs,
            'gt_img': gt,
        }
        data = getattr(self, f'{self.mode}_pipeline')(data)
        return data

    def collate_batch(self, data_list):
        data_dict = {}
        for k in data_list[0].keys():
            if k == 'gt_img' and len(data_list[0][k]) == 0:
                continue
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