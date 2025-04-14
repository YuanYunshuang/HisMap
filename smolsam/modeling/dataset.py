import os
import glob
from random import random

import cv2
import torch
import numpy as np
import json
from torch.utils.data import Dataset
try:
    from torchvision.transforms import v2
except:
    pass


class MapDataset(Dataset):
    def __init__(self, data_root, mode="train"):
        self.data_root = data_root
        self.mode = mode

        with open(os.path.join(self.data_root, "dinfo.json")) as f:
            self.stats = json.load(f)
        with open(os.path.join(self.data_root, f"{mode}.txt")) as f:
            self.file_list = [x.strip() for x in f.readlines()]
        self.transform = MapTransform()
        print(f"number of images: {len(self.file_list)}")
        self.get_loading_list()



    def get_loading_list(self):
        loading_list_src = {}
        for name, info in self.stats.items():
            key = name.rsplit("_", 2)[0]
            if key not in loading_list_src:
                loading_list_src[key] = {i: [] for i in range(1, 5)}
            for k, v in info.items():
                if int(k) == 0:
                    continue
                if 0.1 < v < 0.9 :
                    loading_list_src[key][int(k)].append(name)
        self.loading_list = loading_list_src

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        tgt_image_name = self.file_list[index]
        loc, tgt_name = tgt_image_name.split(".")
        tgt_map_key = tgt_image_name.rsplit("_", 2)[0]
        tgt_info = self.stats[tgt_image_name]
        # available classes of the current target image
        tgt_cls = [int(k) for k, v in tgt_info.items() if v > 0.1 and int(k) > 0]

        if len(tgt_cls) > 0 and (np.random.random() < 0.75 or len(tgt_cls) == 4):
            # half probability to select a cls that is present in both src and tgt image
            selected_cls = np.random.choice(tgt_cls)
        else:
            # half probability to select a cls that is present in src image but not present in tgt image
            not_tgt_cls = [x for x in range(1, 5) if x not in tgt_cls and len(self.loading_list[tgt_map_key][x]) > 0]
            if len(not_tgt_cls) == 0:
                not_tgt_cls = [int(k) for k, v in self.loading_list[tgt_map_key].items() if len(v) > 0]
            selected_cls = np.random.choice(not_tgt_cls)

        src_img_name = np.random.choice(self.loading_list[tgt_map_key][selected_cls])
        _, src_name = src_img_name.split(".")

        src_img = cv2.imread(os.path.join(self.data_root, loc, 'imgs', src_name + '.png'), cv2.IMREAD_UNCHANGED)
        src_lbl = cv2.imread(os.path.join(self.data_root, loc, 'lbls', src_name + '.png'), cv2.IMREAD_UNCHANGED)
        tgt_img = cv2.imread(os.path.join(self.data_root, loc, 'imgs', tgt_name + '.png'), cv2.IMREAD_UNCHANGED)
        tgt_lbl = cv2.imread(os.path.join(self.data_root, loc, 'lbls', tgt_name + '.png'), cv2.IMREAD_UNCHANGED)

        src_lbl = src_lbl == selected_cls
        tgt_lbl = tgt_lbl == selected_cls
        # self.visualize(img, lbl)
        src_img, src_lbl = self.transform(src_img, src_lbl)
        tgt_img, tgt_lbl = self.transform(tgt_img, tgt_lbl)
        # self.visualize(src_img, src_lbl, tgt_img, tgt_lbl)
        src_lbl = src_lbl[::4, ::4]
        # tgt_lbl = tgt_lbl[::4, ::4]
        aux = {'src_name': src_img_name, 'tgt_name': tgt_image_name}
        # print(src_img_name, tgt_image_name)
        return src_img, src_lbl, tgt_img, tgt_lbl, aux

    def visualize(self, src_img, src_lbl, tgt_img, tgt_lbl):
        cv2.imshow('src_img', src_img.numpy().transpose(1, 2, 0))
        cv2.imshow('src_lbl', src_lbl.astype(np.uint8) * 255)
        cv2.imshow('tgt_img', tgt_img.numpy().transpose(1, 2, 0))
        cv2.imshow('tgt_lbl', tgt_lbl.astype(np.uint8) * 255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def collate_fn(batch):
        src_img, src_lbl, tgt_img, tgt_lbl, aux = zip(*batch)
        src_img = torch.stack(src_img, dim=0)
        src_lbl = torch.stack(src_lbl, dim=0)
        tgt_img = torch.stack(tgt_img, dim=0)
        tgt_lbl = torch.stack(tgt_lbl, dim=0)
        return src_img, src_lbl, tgt_img, tgt_lbl, aux


class MapTransform:
    def __init__(self, rot_degrees=30, color_jitter=0.05):
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
            v2.ToDtype(torch.bool, scale=False),
        ])

    def __call__(self, img, lbl):
        img = self.img_transforms(img)
        lbl = self.lbl_transforms(lbl)
        img, lbl = self.geometric_transforms(img, lbl)
        return img, lbl.squeeze()



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


def generate_meta_info(data_root, merge_water=True):
    cities = [x for x in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, x))]
    stats = {}
    train_split = []
    test_split = []
    for city in cities:
        lbl_path = os.path.join(data_root, city, "lbls")
        file_list = sorted(os.listdir(lbl_path))

        for f in file_list:
            name = f"{city}.{f.split('.')[0]}"
            lbl = cv2.imread(os.path.join(lbl_path, f), cv2.IMREAD_UNCHANGED)
            if merge_water:
                lbl[lbl==5] = 4
            stat = {}
            for i in range(5):
                stat[i] = int(np.sum(lbl == i)) / lbl.size
            stats[name] = stat

            x, y = f.split(".")[0].split("_")[-2:]
            if ((int(x) + int(y)) / 384) % 2 == 1:
                train_split.append(name)
            else:
                test_split.append(name)

    with open(os.path.join(data_root, "dinfo.json"), "w") as fh:
        json.dump(stats, fh, indent=4)
    with open(os.path.join(data_root, "train.txt"), "w") as fh:
        fh.write("\n".join(train_split))
    with open(os.path.join(data_root, "test.txt"), "w") as fh:
        fh.write("\n".join(test_split))
    print(f"Found {len(stats)} samples, including {len(train_split)} train samples and {len(test_split)} test samples.")
    return stats

if __name__ == '__main__':
    generate_meta_info("/home/yuan/data/HisMap/SMOLdata")
