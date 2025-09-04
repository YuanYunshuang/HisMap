import os

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
    def __init__(self, data_root, mode="train", info_file=None, fix_src_lbl=0, fg_ratio=0.01):
        """
        Initializes an instance of the class with dataset information, loads file lists,
        and initializes transformations.

        Attributes:
        data_root: str
            The root directory of the dataset containing all related files.
        mode: str
            Indicates the mode in which the dataset is being used. Default is "train".
        info_file: str
            Path to the json file containing the dataset information.
        fix_src_lbl: bool
            A flag used to determine whether source labels should be fixed. Default is 0:not fixed.
            Other options: 1: fixed to 1, 2: fixed to 2, 3: fixed to 3, 4: fixed to 4.
        stats: dict
            Contains the dataset statistics loaded from the 'dinfo.json' file in the
            data_root.
        file_list: list[str]
            A list containing file names from the specified file in the current mode.
        transform: MapTransform
            Instance of the MapTransform class used for applying transformations.

        Parameters:
        data_root: str
            Root directory where dataset files are stored.
        mode: str, optional
            Dataset mode, either 'train', 'test', etc. Default is "train".
        fix_src_lbl: bool, optional
            Whether to fix source labels. Default is 0:not fixed.
            Other options: 1: fixed to 1, 2: fixed to 2, 3: fixed to 3, 4: fixed to 4.

        """
        self.data_root = data_root
        self.mode = mode
        self.fix_src_lbl = fix_src_lbl
        self.fg_ratio = fg_ratio

        if info_file is None:
            info_file = "dinfo_fewshot.json" if mode == "fewshot" else "dinfo.json"
        else:
            info_file = f"dinfo_{info_file}.json"
        with open(os.path.join(self.data_root, info_file)) as f:
            self.stats = json.load(f)
        with open(os.path.join(self.data_root, f"{mode}.txt")) as f:
            self.file_list = [x.strip() for x in f.readlines()]
            self.stats = {k: v for k, v in self.stats.items() if k in self.file_list}

        self.transform = MapTransformTrain() if 'train' in mode else MapTransformTest()
        print(f"number of images: {len(self.file_list)}")
        self.get_loading_list()

    def get_loading_list(self):
        loading_list_src = {}
        for name, info in self.stats.items():
            key = name.rsplit("_", 2)[0]
            if key not in loading_list_src:
                if self.fix_src_lbl > 0:
                    loading_list_src[key] = {self.fix_src_lbl: []}
                else:
                    loading_list_src[key] = {i: [] for i in range(1, 5)}
            for k, v in info.items():
                if int(k) == 0:
                    continue
                if 0 < self.fix_src_lbl != int(k):
                    continue
                if self.fg_ratio < v < 0.9 :
                    loading_list_src[key][int(k)].append(name)

        # omit evaluation for map sheets that does not contain fixed target class
        if self.fix_src_lbl > 0:
            invalid_map_key = []
            for k, v in loading_list_src.items():
                if len(v[self.fix_src_lbl]) == 0:
                    invalid_map_key.append(k)
            for k in invalid_map_key:
                del loading_list_src[k]

            total_file_num = len(self.file_list)
            self.file_list = [n for n in self.file_list if n.rsplit('_', 2)[0] not in invalid_map_key]
            print(f"{len(self.file_list)} images out of {total_file_num} are selected for evaluation of the fixed src label {self.fix_src_lbl}.")
        self.loading_list = loading_list_src

    def select_tgt_cls(self, tgt_info, tgt_map_key):
        # available classes of the current target image
        tgt_cls = [int(k) for k, v in tgt_info.items() if v > self.fg_ratio and int(k) > 0]

        if len(tgt_cls) > 0 and (np.random.random() < 0.75 or len(tgt_cls) == 4):
            # p probability to select a cls that is present in both src and tgt image
            selected_cls = np.random.choice(tgt_cls)
        else:
            # p probability to select a cls that is present in src image but not present in tgt image
            not_tgt_cls = [x for x in range(1, 5) if x not in tgt_cls and len(self.loading_list[tgt_map_key][x]) > 0]
            if len(not_tgt_cls) == 0:
                not_tgt_cls = [int(k) for k, v in self.loading_list[tgt_map_key].items() if len(v) > 0]
            selected_cls = np.random.choice(not_tgt_cls)
        return selected_cls

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        tgt_image_name = self.file_list[index]
        loc, tgt_name = tgt_image_name.split(".")
        tgt_map_key = tgt_image_name.rsplit("_", 2)[0]
        tgt_info = self.stats[tgt_image_name]

        if self.fix_src_lbl > 0:
            selected_cls = self.fix_src_lbl
        else:
            selected_cls = self.select_tgt_cls(tgt_info, tgt_map_key)

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
        cv2.imshow('src_lbl', src_lbl.numpy().astype(np.uint8) * 255)
        cv2.imshow('tgt_img', tgt_img.numpy().transpose(1, 2, 0))
        cv2.imshow('tgt_lbl', tgt_lbl.numpy().astype(np.uint8) * 255)
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


class MapTransformTrain:
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


class MapTransformTest:
    def __init__(self):
        self.img_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.lbl_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.bool, scale=False),
        ])

    def __call__(self, img, lbl=None):
        img = self.img_transforms(img)
        if lbl is not None:
            lbl = self.lbl_transforms(lbl).squeeze()
            return img, lbl
        else:
            return img



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
    # 50/50
    with open(os.path.join(data_root, "train50.txt"), "w") as fh:
        fh.write("\n".join(train_split))
    with open(os.path.join(data_root, "test50.txt"), "w") as fh:
        fh.write("\n".join(test_split))

    # 75 / 25
    total_file_num = len(train_split) + len(test_split)
    goal_num = int(total_file_num * 0.75)
    selected = np.random.choice(test_split, goal_num - len(train_split), replace=False)
    train_split.extend(selected)
    test_split = [x for x in test_split if x not in selected]
    with open(os.path.join(data_root, "train75.txt"), "w") as fh:
        fh.write("\n".join(train_split))
    with open(os.path.join(data_root, "test25.txt"), "w") as fh:
        fh.write("\n".join(test_split))

    print(f"Found {len(stats)} samples.")
    return stats

if __name__ == '__main__':
    generate_meta_info("/home/yuan/data/HisMap/SMOLdata")
