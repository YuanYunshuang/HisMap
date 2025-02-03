import glob
import os
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset.pipeline import Pipeline
from src.tools.map_analyzer import read_tiff

CLASSES = ['Woodland', 'Grassland', 'Settlement', 'FlowingWater', 'StandingWater']


def generate_dataset_v1(path_in, path_out, mode='train', crop_size=384, num_pixel_present=100, margin=128):
    """
    Split train and test set according to the given mode, the map sheets in the corresponding folder will be cropped
    and regarded as the same split.
    """
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
        h, w = label.shape[0], label.shape[1]
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


def generate_dataset_v2(path_in, path_out, mode='train', crop_size=384, num_pixel_present=100,
                        margin=128, add_gray_img=False):
    """
    Split train and test set patch-wise. On each map sheet, the cropped patches are alternately
    saved to the train and test sets.
    """
    if mode == 'test': margin = None
    if margin is None: margin = crop_size
    label_files = glob.glob(os.path.join(path_in, '*/label/*.png'))
    path_out = os.path.join(path_out, mode)
    os.makedirs(path_out, exist_ok=True)
    for f in label_files:
        path, basename = os.path.split(f)
        split = path.split('/')[-2]
        patch, year = basename.split('.')[0].split('_')
        label = cv2.imread(f, cv2.IMREAD_GRAYSCALE) / 50
        filename = os.path.join(path_in, split, 'maps', f"{patch}_{year}.tif")
        if not os.path.exists(filename):
            filename = os.path.join(path_in, split, 'label', f"{patch}_{int(year) - 1:d}.tif")
        if not os.path.exists(filename):
            filename = os.path.join(path_in, split, 'label', f"{patch}_{int(year) + 1:d}.tif")
        if not os.path.exists(filename):
            continue
        print(filename)
        hismap = read_tiff(filename)
        # if hismap.ndim == 3:
        #     hismap = cv2.cvtColor(hismap, cv2.COLOR_BGR2GRAY)

        h, w = label.shape[0], label.shape[1]
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
            # M = cv2.cvtColor(M, cv2.COLOR_BGR2GRAY)

            l_cls = ''
            for i in range(1, 5):
                if i == 4 and np.sum(L >= i) > num_pixel_present:
                    l_cls += '1'
                elif np.sum(L == i) > num_pixel_present:
                    l_cls += '1'
                else:
                    l_cls += '0'
            if M.ndim == 2:
                M = M[:, :, None].repeat(3, axis=-1)
            M = np.pad(M, ((0, crop_size - M.shape[0]), (0, crop_size - M.shape[1]), (0, 0)), mode='constant', constant_values=0)

            cv2.imwrite(os.path.join(path_out, f"{patch}_{year}_{x1}_{y1}_{l_cls}_a.jpg"), M)
            if add_gray_img and  mode == 'train' and hismap.ndim == 3:
                M = cv2.cvtColor(M, cv2.COLOR_BGR2GRAY)[:, :, None].repeat(3, axis=-1)
                cv2.imwrite(os.path.join(path_out, f"{patch}_{year}_{x1}_{y1}_{l_cls}_b.jpg"), M)


def generate_dataset_v3(path_in, path_out, mode='train', crop_size=384, num_pixel_present=100,
                        margin=192):
    """
    Only use data before 2000
    """
    if mode == 'test': margin = None
    if margin is None or margin==0: margin = crop_size
    label_files = glob.glob(os.path.join(path_in, '*/label/*.png'))
    path_out = os.path.join(path_out, mode)
    os.makedirs(path_out, exist_ok=True)
    for f in label_files:
        path, basename = os.path.split(f)
        split = path.split('/')[-2]
        patch, year = basename.split('.')[0].split('_')
        if int(year) > 2000:
            continue
        label = cv2.imread(f, cv2.IMREAD_GRAYSCALE) / 50
        filename = os.path.join(path_in, split, 'maps', f"{patch}_{year}.tif")
        if not os.path.exists(filename):
            continue
        print(filename)
        hismap = read_tiff(filename)
        # if hismap.ndim == 3:
        #     hismap = cv2.cvtColor(hismap, cv2.COLOR_BGR2GRAY)

        h, w = label.shape[0], label.shape[1]
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
            # M = cv2.cvtColor(M, cv2.COLOR_BGR2GRAY)

            l_cls = ''
            for i in range(1, 5):
                if i == 4 and np.sum(L >= i) > num_pixel_present:
                    l_cls += '1'
                elif np.sum(L == i) > num_pixel_present:
                    l_cls += '1'
                else:
                    l_cls += '0'
            if M.ndim == 2:
                M = M[:, :, None].repeat(3, axis=-1)
            M = np.pad(M, ((0, crop_size - M.shape[0]), (0, crop_size - M.shape[1]), (0, 0)), mode='constant', constant_values=0)
            M = cv2.cvtColor(M, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(path_out, f"{patch}_{year}_{x1}_{y1}_{l_cls}.jpg"), M)



class LglnclsDataset(Dataset):

    def __init__(self,
                 cfgs,
                 mode='train',
    ):
        self.cfgs = cfgs
        self.root_dir = cfgs['data_path']
        self.mode = mode
        self.multi_cls = cfgs.get('multi_cls', False)
        if not self.multi_cls:
            assert 'cls_label' in cfgs
            self.cls_label = cfgs['cls_label']
        label_pattern = cfgs.get('label_pattern', None)

        if mode == 'inference':
            self.samples = []
            self.img_dir = self.root_dir
            for split in ['train', 'test']:
                self.samples.extend([os.path.join(split, x) for x in os.listdir(os.path.join(self.root_dir, split))])
        else:
            self.img_dir = os.path.join(self.root_dir, self.mode)
            self.samples = os.listdir(self.img_dir)
        if label_pattern is not None:
            samples = []
            if len(label_pattern) == 5:
                assert len(label_pattern) == len(self.samples[0].split('_')[-1].split('.')[0])
            else:
                assert len(label_pattern) == len(self.samples[0].split('_')[-2])
            for sample in self.samples:
                label = sample[:-4].split('_')[4]
                flag = True
                for pi in [i for i, p in enumerate(label_pattern) if p != '*']:
                    if label[pi] != label_pattern[pi]:
                        flag = False
                        break
                if flag:
                    samples.append(sample)
            self.samples = samples
        pl_name = 'train' if mode == 'train' else 'test'
        self.pipeline = Pipeline(cfgs[f'{pl_name}_pipeline'])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        filename = self.samples[item]
        patch, year, x , y, l = filename.split('.')[0].split('_')[:5]
        img = cv2.imread(os.path.join(self.img_dir, filename), cv2.IMREAD_UNCHANGED)
        if self.multi_cls:
            label = torch.Tensor([int(x) for x in l])
        else:
            if self.cls_label < 3 or len(l) == 4:
                label = int(l[self.cls_label])
            else:
                label = int(int(l[3]) or int(l[4]))

        data = {
            'patch': patch,
            'year': year,
            'x': x,
            'y': y,
            'img': img,
            'label': label,
        }
        data = self.pipeline(data)
        return data

    @staticmethod
    def collate_batch(data_list):
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

    # for mode in ['train', 'test']:
    #     generate_dataset_v2(
    #         args.path_in,
    #         args.path_out,
    #         mode=mode,
    #         crop_size=args.crop_size,
    #     )

    generate_dataset_v3(
        args.path_in,
        args.path_out,
        mode='train',
        crop_size=args.crop_size,
        margin=0
    )

    # from src.utils.misc import load_yaml
    # cfg = load_yaml('../config/cvit/demo.yaml')
    # data_cfg = cfg['DATASET']
    # dataset = LglnclsDataset(data_cfg, mode='train')
    # dataloader = DataLoader(dataset, batch_size=2, num_workers=4, collate_fn=dataset.collate_batch)
    # for d in dataloader:
    #     print(d)