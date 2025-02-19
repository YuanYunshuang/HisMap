import os
import glob
from random import random

import cv2
import torch
import numpy as np
from geotiff import GeoTiff
from torch.utils.data import Dataset
try:
    from torchvision.transforms import v2
except:
    pass


class MapDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.lbl_path = os.path.join(data_root, "lbls")
        self.img_path = os.path.join(data_root, "imgs")
        self.lbl_path_files = sorted(os.listdir(self.lbl_path))
        self.transform = MapTransform()
        print(f"number of images: {len(self.lbl_path_files)}")

    def __len__(self):
        return len(self.lbl_path_files)

    def __getitem__(self, index):
        name = os.path.basename(self.lbl_path_files[index])
        img = cv2.imread(os.path.join(self.img_path, name), cv2.IMREAD_UNCHANGED)
        if img.shape[0] != 1024 or img.shape[1] != 1024:
            print(f"Image shape {img.shape}")
        lbl = cv2.imread(os.path.join(self.lbl_path, name), cv2.IMREAD_UNCHANGED)
        # self.visualize(img, lbl)
        img, lbl = self.transform(img, lbl)
        # self.visualize(img.numpy().transpose(1, 2, 0), lbl.numpy().astype(np.uint8))
        return img, lbl

    def visualize(self, img, lbl):
        cv2.imshow('img', img)
        cv2.imshow('lbl', lbl)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def collate_fn(batch):
        img, lbl = zip(*batch)
        img = torch.stack(img, dim=0)
        lbl = torch.stack(lbl, dim=0)
        return img, lbl


class MapTransform:
    def __init__(self, rot_degrees=30, color_jitter=0.2):
        self.geometric_transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=(-rot_degrees, rot_degrees)),
        ])

        self.img_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            v2.ColorJitter(color_jitter, color_jitter, color_jitter),
        ])

        self.lbl_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.long, scale=False),
        ])

    def __call__(self, img, lbl):
        img = self.img_transforms(img)
        lbl = self.lbl_transforms(lbl)
        img, lbl = self.geometric_transforms(img, lbl)
        return img, lbl.squeeze()


def read_tiff(filename):
    if not os.path.exists(filename):
        print('File not found: {}'.format(filename))
        return None
    geo_tiff = GeoTiff(filename)
    tiff_array = np.array(geo_tiff.read())
    return tiff_array


def generate_source_dataset(path_in, path_out, scale=1.0, img_size=1024, margin=128):
    img_dir = os.path.join(path_in, 'maps')
    lbl_dir = os.path.join(path_in, 'label')
    os.makedirs(os.path.join(path_out, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(path_out, 'lbls'), exist_ok=True)

    lbl_files = os.listdir(lbl_dir)
    for lbl_file in lbl_files:
        name, ext = os.path.splitext(lbl_file)
        TIF = read_tiff(os.path.join(img_dir, f'{name}.tif'))
        LABEL = cv2.imread(os.path.join(lbl_dir, lbl_file), cv2.IMREAD_UNCHANGED) // 50

        if scale != 1.0:
            TIF = cv2.resize(TIF, None, fx=scale, fy=scale)
            LABEL = cv2.resize(LABEL, None, fx=scale, fy=scale)
        if TIF.ndim == 3:
            TIF = TIF[..., ::-1]

        H, W = min(TIF.shape[0], LABEL.shape[0]), min(TIF.shape[1], LABEL.shape[1])
        x = np.arange(0, W, img_size-margin)
        y = np.arange(0, H, img_size-margin)
        x[-1] = W - img_size
        y[-1] = H - img_size
        yy, xx = np.meshgrid(x, y)
        xys = np.vstack((xx.flatten(), yy.flatten())).T

        for xy in xys:
            img = TIF[xy[0]:xy[0] + img_size, xy[1]:xy[1] + img_size]
            lbl = LABEL[xy[0]:xy[0] + img_size, xy[1]:xy[1] + img_size]
            # lbl = lbl.astype(np.int16) - 1
            lbl[lbl == 0] = 255
            # lbl = lbl.astype(np.uint8)

            img = np.pad(img, ((0, img_size - img.shape[0]), (0, img_size - img.shape[1]), (0, 0)),
                         'constant', constant_values=0)
            lbl = np.pad(lbl, ((0, img_size - lbl.shape[0]), (0, img_size - lbl.shape[1])),
                         'constant', constant_values=0)

            cv2.imwrite(os.path.join(path_out, 'imgs', f'{name}_{xy[0]}_{xy[1]}.png'), img)
            cv2.imwrite(os.path.join(path_out, 'lbls', f'{name}_{xy[0]}_{xy[1]}.png'), lbl)


def generate_annotation_data(path_in, path_out, scale=1.0, img_size=1024):
    img = read_tiff("/home/data/HisMAP/Hameln/TK_projektiv_entzerrt_1m/3821_1898.tif")
    name = "3821_1898"
    H, W = img.shape[0], img.shape[1]
    for x in range(0, W, img_size):
        for y in range(0, H, img_size):
            img_crop = img[y:y + img_size, x:x + img_size]
            cv2.imwrite(os.path.join(path_out, f'{name}_{x}_{y}.png'), img_crop)


def get_dataloader(root_dir, batch_size=4, mode='train'):
    dataset = MapDataset(root_dir)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             sampler=None,
                                             num_workers=4,
                                             shuffle=True,
                                             collate_fn=dataset.collate_fn,
                                             pin_memory=True,
                                             drop_last=True)
    return dataloader


if __name__ == '__main__':
    # 1. Generate pre-training dataset
    # generate_source_dataset(
    #     "/home/yuan/data/HisMap/LGLN/dataset/train",
    #     "/home/yuan/data/HisMap/ADA/pre_train",
    #     1.0
    # )

    # 2. Test dataset
    # dataloader = get_dataloader("/home/yuan/data/HisMap/ADA/pre_train")
    # for data in dataloader:
    #     pass

    # 3. Generate annotation data
    generate_annotation_data(
        "",
        "/home/data/HisMAP/anno_data"
    )