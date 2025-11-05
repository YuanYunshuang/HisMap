import os

from PIL import Image
import torch
import numpy as np
import json
from torch.utils.data import Dataset
import random
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import InterpolationMode
from typing import List
try:
    from torchvision.transforms import v2
except:
    pass


def get_dataset(dataset_name, **kwargs):
    if dataset_name == "smol":
        return SMOLDataset(**kwargs)
    elif dataset_name == "syntra":
        return SynTraDataset(**kwargs)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")


class SMOLDataset(Dataset):
    INVALID_CLS = ['background', 'nonlabeled']
    def __init__(self, data_root, num_notions=1, num_src=1, mode="train", 
                 info_file=None, fg_ratio=0.05, pos_prob=0.75, filtering_keys=None,
                 src_info_file=None, src_cls=None, **kwargs):
        """
        Initializes an instance of the class with dataset information, loads file lists,
        and initializes transformations.
        Parameters:
        data_root: str
            The root directory of the dataset containing all related files.
        mode: str
            Indicates the mode in which the dataset is being used. Default is "train".
        info_file: str
            Path to the json file containing the data sample information.
        fg_ratio: float, optional
            Foreground ratio threshold for source masks. Default is 0.1.
        pos_prob: float, optional
            Probability of selecting a class that is present in both source and target images. Default is 0.75.
        filtering_keys: str, optional
            Comma-separated keys to filter the dataset files. Default is None.
        """
        self.data_root = data_root
        self.num_notions = num_notions
        self.num_src = num_src
        self.mode = mode
        self.fg_ratio = fg_ratio
        self.pos_prob = pos_prob
        
        if mode == "test":
            assert src_info_file is not None, "In test mode, src_info_file must be provided."
            assert src_cls is not None, "In test mode, src_cls must be provided."
            
        self.fixed_cls = src_cls
        self.datasets = os.listdir(self.data_root)
        # load samples
        self.label_ratios = {}
        self.label_colors = {}
        self.file_list = []
        self.src_file_list = []
        available_classes = set()
        for ds in self.datasets:
            if filtering_keys is not None and not any([k in ds for k in filtering_keys]):
                continue
            with open(os.path.join(self.data_root, ds, info_file)) as f:
                lines = list(json.load(f).keys())
                self.file_list.extend(lines)
            with open(os.path.join(self.data_root, ds, f"{ds}.json")) as f:
                self.label_ratios.update(json.load(f))
            with open(os.path.join(self.data_root, ds, f"color_map.json")) as f:
                self.label_colors[ds] = json.load(f)
                available_classes.update(set(self.label_colors[ds].keys()) - set(self.INVALID_CLS))
            if src_info_file is not None:
                with open(os.path.join(self.data_root, ds, src_info_file)) as f:
                    src_lines = list(json.load(f).keys())
                    self.src_file_list.extend(src_lines)
        
        if self.fixed_cls is not None:
            assert self.fixed_cls in available_classes, \
                f"Fixed source class {self.fixed_cls} is not in available classes {available_classes} of the current loaded dataset."

        if len(self.src_file_list) == 0:
            self.src_file_list = self.file_list.copy()

        if filtering_keys is not None:
            filtering_keys = filtering_keys.split(",")
            filtered_file_list = []
            filtered_src_file_list = []
            for fname in self.file_list:
                if any([k in fname for k in filtering_keys]):
                    filtered_file_list.append(fname)
            for fname in self.src_file_list:
                if any([k in fname for k in filtering_keys]):
                    filtered_src_file_list.append(fname)
            self.file_list = filtered_file_list
            self.src_file_list = filtered_src_file_list

        self.file_list = list(set(self.file_list))
        self.src_file_list = list(set(self.src_file_list))
        self.transform = MapTransformTrain() if 'train' in mode or 'nshot' in mode else MapTransformTest()
        print(f"Number of {mode} images: {len(self.file_list)}")
        self.get_loading_list()

    def get_loading_list(self):
        loading_list_src = {}
        for name in self.src_file_list:
            info = self.label_ratios[name]
            key = name.rsplit(".", 1)[0] # subdataset key (map sheet key)
            if key not in loading_list_src:
                loading_list_src[key] = {c: [] for c in info.keys() if c not in self.INVALID_CLS}
            for k, v in info.items(): # k: class, v: ratio
                if k in self.INVALID_CLS:
                    continue
                if self.fg_ratio < v < 0.9 :
                    loading_list_src[key][k].append(name)
        if self.mode == "test":
            file_list = [f for f in self.file_list if len(loading_list_src[f.rsplit(".", 1)[0]][self.fixed_cls]) > 0]
            if len(self.file_list) != len(file_list):
                print(f"Warning: {len(self.file_list) - len(file_list)} target images do not have available source images and will be removed for testing.")
                self.file_list = file_list
        self.loading_list = loading_list_src

    def select_tgt_cls(self, tgt_full_name):
        if 'railway' in tgt_full_name:
            fg_ratio = 0.01
        else:
            fg_ratio = self.fg_ratio

        # available classes of the current target image
        tgt_info = self.label_ratios[tgt_full_name]
        dataset = tgt_full_name.rsplit(".", 1)[0]
        tgt_cls = [k for k, v in tgt_info.items() if v > fg_ratio and k not in self.INVALID_CLS]

        if len(tgt_cls) > 0 and (np.random.random() < self.pos_prob or \
                                 len(tgt_cls) == len(tgt_info) - len(self.INVALID_CLS)):
            # p probability to select a cls that is present in both src and tgt image
            selected_cls = np.random.choice(tgt_cls)
        else:
            # p probability to select a cls that is present in src image but not present in tgt image
            not_tgt_cls = [x for x in tgt_info.keys() if x not in (tgt_cls + self.INVALID_CLS) and len(self.loading_list[dataset][x]) > 0]
            if len(not_tgt_cls) == 0:
                not_tgt_cls = [k for k, v in self.loading_list[dataset].items() if len(v) > 0 and k not in self.INVALID_CLS]
            selected_cls = np.random.choice(not_tgt_cls)

        return selected_cls

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        tgt_full_name = self.file_list[index]
        dataset, _ = tgt_full_name.rsplit(".", 1)

        if self.mode == "train":
            selected_cls = self.select_tgt_cls(tgt_full_name) 
        else:
            selected_cls = self.fixed_cls  
        selected_colors = [self.label_colors[dataset][selected_cls] if selected_cls is not None else None]      
        available_src = self.loading_list[dataset][selected_cls]
        if len(available_src) == 0:
            print(f"Warning: No available source images for target image {tgt_full_name} and selected class {selected_cls}.")
        src_full_names = np.random.choice(available_src, self.num_src, replace=len(available_src)<self.num_src).tolist()           

        # load images and labels
        imgs = []
        lbls = []
        for img_full_name in [tgt_full_name] + src_full_names:
            ds, img_name = img_full_name.rsplit(".", 1)
            img = Image.open(os.path.join(self.data_root, ds, 'imgs', img_name + '.png'))
            img = np.array(img)
            lbl = Image.open(os.path.join(self.data_root, ds, 'lbls', img_name + '.png'))
            # get selected color idx
            palette = lbl.getpalette()
            palette = np.array(palette).reshape(-1, 3).tolist()
            selected_color_inds = [palette.index(color) if color is not None else None for color in selected_colors]
            assert 0 < len(selected_color_inds) <= self.num_notions, \
                f"Number of selected notions {len(selected_color_inds)} should be > 0 and < {self.num_notions}."
            multi_notion_lbl = np.zeros(lbl.size + (self.num_notions,), dtype=bool)
            lbl = np.array(lbl.convert("P"))
            for i, color_idx in enumerate(selected_color_inds):
                if color_idx is not None:
                    multi_notion_lbl[..., i] = lbl == color_idx
            imgs.append(img)
            lbls.append(multi_notion_lbl)

        # self.visualize(img, lbl)
        imgs, lbls = self.transform(imgs, lbls)
        # self.visualize(src_img, src_lbl, tgt_img, tgt_lbl)
        src_imgs = torch.stack(imgs[1:], 0)
        tgt_img = imgs[0]
        src_lbls = torch.stack(lbls[1:], 0)
        tgt_lbl = lbls[0]
        src_lbls = src_lbls[..., ::4, ::4]
        # tgt_lbl = tgt_lbl[::4, ::4]
        aux = {'src_name': src_full_names, 'tgt_name': tgt_full_name}
        # print(src_img_name, tgt_image_name)
        return src_imgs, src_lbls, tgt_img, tgt_lbl, aux

    def subdataset_id(self, index: int) -> int:
        # Derive subdataset id from filename prefix before the first '.'
        # Example: "locA.some_image_name_001.png" -> "locA"
        def get_loc(fname):
            if 'siegfried' in fname:
                loc = fname.split(".")[1].split('_')[0]
            else:
                loc = fname.split('.')[0]
            return loc
        fname = self.file_list[index]
        loc = get_loc(fname)
        # Map loc strings to stable ids [0..K-1]
        if not hasattr(self, "_loc2id"):
            # build mapping once
            uniq: List[str] = sorted({get_loc(f) for f in self.file_list})
            self._loc2id = {v: i for i, v in enumerate(uniq)}
        return self._loc2id[loc]

    @staticmethod
    def collate_fn(batch):
        src_img, src_lbl, tgt_img, tgt_lbl, aux = zip(*batch)
        src_img = torch.stack(src_img, dim=0)
        src_lbl = torch.stack(src_lbl, dim=0)
        tgt_img = torch.stack(tgt_img, dim=0)
        tgt_lbl = torch.stack(tgt_lbl, dim=0)
        
        return src_img, src_lbl, tgt_img, tgt_lbl, aux
    


class SynTraDataset(Dataset):
    INVALID_CLS = ['background', 'nonlabeled']
    def __init__(self, data_root, info_file, num_notions=1, num_src=1, mode="train", 
                 fg_ratio=0.05, pos_prob=0.75, filtering_keys=None, **kwargs):
        """
        Initializes an instance of the class with dataset information, loads file lists,
        and initializes transformations.

        Parameters:
        data_root: str
            The root directory of the dataset containing all related files.
        info_file: str
            Path to the json file containing the dataset information.
        mode: str
            Indicates the mode in which the dataset is being used. Default is "train".
        fg_ratio: float, optional
            Foreground ratio threshold for source masks. Default is 0.1.
        pos_prob: float, optional
            Probability of selecting a class that is present in both source and target images. Default is 0.75.
        filtering_keys: str, optional
            Comma-separated keys to filter the sub-datasets that contain the given keys. Default is None to select all.
        """
        self.data_root = data_root
        self.num_notions = num_notions
        self.num_src = num_src
        self.mode = mode
        self.fg_ratio = fg_ratio
        self.pos_prob = pos_prob

        self.datasets = os.listdir(self.data_root)
        # load samples
        self.src_pool = {}
        self.label_ratios = {}
        self.label_colors = {}
        filtering_keys = filtering_keys.split(",")
        for ds in self.datasets:
            if filtering_keys is not None and not any([k in ds for k in filtering_keys]):
                continue
            if not os.path.isfile(os.path.join(self.data_root, ds, info_file)):
                Warning(f"Info file {info_file} does not exist in dataset folder {ds}. Skip this dataset.")
                continue
            with open(os.path.join(self.data_root, ds, info_file)) as f:
                self.src_pool.update(json.load(f))
            with open(os.path.join(self.data_root, ds, f"{ds}.json")) as f:
                self.label_ratios.update(json.load(f))
            with open(os.path.join(self.data_root, ds, f"color_map.json")) as f:
                self.label_colors[ds] = json.load(f)

        self.file_list = list(self.src_pool.keys())

        self.transform = MapTransformTrain() if 'train' in mode or 'nshot' in mode else MapTransformTest()
        print(f"Number of {mode} images: {len(self.file_list)}")

    def select_tgt_cls(self, tgt_full_name, src_full_names):
        if self.mode == "test":
            fg_ratio = 0.0
        elif 'railway' in tgt_full_name:
            fg_ratio = 0.01
        else:
            fg_ratio = self.fg_ratio
        # parse available classes of the current target image
        tgt_info = self.label_ratios[tgt_full_name]
        tgt_available_classes = [k for k, v in tgt_info.items() if v > fg_ratio and k not in self.INVALID_CLS]
        tgt_available_classes = set(tgt_available_classes)

        # parse available classes of the current source images
        src_available_classes = []
        for src_full_name in src_full_names:
            src_info = self.label_ratios[src_full_name]
            src_available_classes.extend([k for k, v in src_info.items() if v > fg_ratio and k not in self.INVALID_CLS])
        src_available_classes = set(src_available_classes)

        # cal sample probability: p to select a class that is present in both src and tgt image
        # prob 1-p to select a class that is present in src image but not present in tgt image
        available_classes = list(src_available_classes)
        pos_classes = list(src_available_classes & tgt_available_classes)
        neg_classes = list(src_available_classes - tgt_available_classes)
        
        if len(available_classes) == 0:
            Warning(f"No available classes for target image {tgt_full_name} and source images {src_full_names}. Return None.")
            return None
        
        if random.random() < self.pos_prob and len(pos_classes) > 0:
            selected_cls = np.random.choice(pos_classes)
        else:
            if len(neg_classes) == 0:
                neg_classes = pos_classes
            selected_cls = np.random.choice(neg_classes)
        
        return selected_cls

    def _select_src_samples(self, tgt_full_name):
        src_full_names = self.src_pool[tgt_full_name]['src_samples']
        src_dists = self.src_pool[tgt_full_name]['distances']
        if self.mode == "train":
            # random select num_src from the available src images
            src_full_names = np.random.choice(src_full_names, self.num_src, replace=len(src_full_names)<self.num_src).tolist()
        else:
            # select top-k closest src images
            sorted_indices = np.argsort(src_dists)
            src_full_names = [src_full_names[i] for i in sorted_indices[:self.num_src]]
        return src_full_names

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        tgt_full_name = self.file_list[index]
        dataset, _ = tgt_full_name.rsplit(".", 1)
        src_full_names = self._select_src_samples(tgt_full_name)

        if self.mode == "train":
            selected_cls = self.select_tgt_cls(tgt_full_name, src_full_names) 
            selected_colors = [self.label_colors[dataset][selected_cls] if selected_cls is not None else None]                 
        else:
            # for test mode, use all notions
            selected_colors = [v for k, v in self.label_colors[dataset].items() if k not in self.INVALID_CLS]
            assert len(selected_colors) <= self.num_notions, \
                f"During test, number of selected notions {len(selected_colors)} should be == {self.num_notions}."

        # load images and labels
        imgs = []
        lbls = []
        for img_full_name in [tgt_full_name] + src_full_names:
            ds, img_name = img_full_name.rsplit(".", 1)
            img = Image.open(os.path.join(self.data_root, ds, 'imgs', img_name + '.png'))
            img = np.array(img)
            lbl = Image.open(os.path.join(self.data_root, ds, 'lbls', img_name + '.png'))
            # get selected color idx
            palette = lbl.getpalette()
            palette = np.array(palette).reshape(-1, 3).tolist()
            selected_color_inds = [palette.index(color) if color is not None else None for color in selected_colors]
            assert 0 < len(selected_color_inds) <= self.num_notions, \
                f"Number of selected notions {len(selected_color_inds)} should be > 0 and < {self.num_notions}."
            multi_notion_lbl = np.zeros(lbl.size + (self.num_notions,), dtype=bool)
            lbl = np.array(lbl.convert("P"))
            for i, color_idx in enumerate(selected_color_inds):
                if color_idx is not None:
                    multi_notion_lbl[..., i] = lbl == color_idx
            imgs.append(img)
            lbls.append(multi_notion_lbl)

        # self.visualize(img, lbl)
        imgs, lbls = self.transform(imgs, lbls)
        # self.visualize(src_img, src_lbl, tgt_img, tgt_lbl)
        src_imgs = torch.stack(imgs[1:], 0)
        tgt_img = imgs[0]
        src_lbls = torch.stack(lbls[1:], 0)
        tgt_lbl = lbls[0]
        src_lbls = src_lbls[..., ::4, ::4]
        # tgt_lbl = tgt_lbl[::4, ::4]
        aux = {'src_name': src_full_names, 'tgt_name': tgt_full_name}
        # print(src_img_name, tgt_image_name)
        return src_imgs, src_lbls, tgt_img, tgt_lbl, aux

    def subdataset_id(self, index: int) -> int:
        # Derive subdataset id from filename prefix before the first '.'
        # Example: "locA.some_image_name_001.png" -> "locA"
        def get_loc(fname):
            if 'siegfried' in fname:
                loc = fname.split(".")[1].split('_')[0]
            else:
                loc = fname.split('.')[0]
            return loc
        fname = self.file_list[index]
        loc = get_loc(fname)
        # Map loc strings to stable ids [0..K-1]
        if not hasattr(self, "_loc2id"):
            # build mapping once
            uniq: List[str] = sorted({get_loc(f) for f in self.file_list})
            self._loc2id = {v: i for i, v in enumerate(uniq)}
        return self._loc2id[loc]

    @staticmethod
    def collate_fn(batch):
        src_img, src_lbl, tgt_img, tgt_lbl, aux = zip(*batch)
        src_img = torch.stack(src_img, dim=0)
        src_lbl = torch.stack(src_lbl, dim=0)
        tgt_img = torch.stack(tgt_img, dim=0)
        tgt_lbl = torch.stack(tgt_lbl, dim=0)
        
        return src_img, src_lbl, tgt_img, tgt_lbl, aux
    

class PadToTargetSize:
    def __init__(self, target_size):
        self.target_size = target_size
    
    def pad_one(self, image, label):
        w, h = image.shape[1:]
        pad_w = max(self.target_size - w, 0)
        pad_h = max(self.target_size - h, 0)

        # Compute padding (left, top, right, bottom)
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top

        # Pad image with 0
        image = F.pad(image, [left, top, right, bottom], fill=0)
        # Pad label with 255
        if label is not None:
            label = F.pad(label, [left, top, right, bottom], fill=False)
        return image, label

    def __call__(self, image, label):
        if isinstance(image, list):
            image_list = []
            label_list = []
            for img, lbl in zip(image, label):
                img, lbl = self.pad_one(img, lbl)
                image_list.append(img)
                label_list.append(lbl)
            return image_list, label_list
        else:
            image, label = self.pad_one(image, label)

        return image, label


class MapTransformTrain:
    def __init__(self, image_size=384, rot_degrees=30, color_jitter=0.01):
        # store params for reproducible geometry sampling per pair
        self.rot_degrees = rot_degrees
        self.img_size = image_size

        self.img_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
            # v2.ColorJitter(color_jitter, color_jitter, color_jitter),
        ])

        self.lbl_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.bool, scale=False),
        ])

        # self.pad_transforms = v2.Compose([
        #     PadToTargetSize(image_size),
        # ])
        self.resize_img = v2.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)
        self.resize_lbl = v2.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST)
    
    def resize(self, imgs, lbls):
        """
        Resize images and labels to target size. Supports either a single image/label
        or lists of them.
        Images use bilinear interpolation and labels use nearest.
        """
        def apply_one(i, l):
            i = self.resize_img(i)
            if l is not None:
                l = self.resize_lbl(l.unsqueeze(0)).squeeze(0)
                try:
                    l = l.to(torch.bool)
                except Exception:
                    # fallback if returned type isn't a tensor
                    pass
            return i, l

        if isinstance(imgs, list):
            img_list = []
            lbl_list = []
            for i, ll in zip(imgs, lbls):
                ii, ll = apply_one(i, ll)
                img_list.append(ii)
                lbl_list.append(ll)
            return img_list, lbl_list
        else:
            return apply_one(imgs, lbls)

    def geometric_transforms(self, img, lbl):
        """
        Sample a single geometric transform (hflip, vflip, rotation angle)
        and apply the same transform to all images and labels. Supports
        either a single image/label or lists of them.
        Images use bilinear interpolation and labels use nearest.
        """
        # sample once per call so all items get identical transform
        do_h = np.random.rand() < 0.5
        do_v = np.random.rand() < 0.5
        angle = float(np.random.uniform(-self.rot_degrees, self.rot_degrees))

        def apply_one(i, l):
            # horizontal flip
            if do_h:
                i = F.hflip(i)
                if l is not None:
                    l = F.hflip(l)
            # vertical flip
            if do_v:
                i = F.vflip(i)
                if l is not None:
                    l = F.vflip(l)

            # rotation: images bilinear, labels nearest
            i = F.rotate(i, angle=angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)
            if l is not None:
                # use nearest to preserve mask values; ensure bool dtype after
                l = F.rotate(l.unsqueeze(0), angle=angle, interpolation=InterpolationMode.NEAREST, 
                             expand=False, fill=False).squeeze(0)
                try:
                    l = l.to(torch.bool)
                except Exception:
                    # fallback if returned type isn't a tensor
                    pass
            return i, l

        if isinstance(img, list):
            imgs = []
            lbls = []
            for i, ll in zip(img, lbl):
                ii, ll = apply_one(i, ll)
                imgs.append(ii)
                lbls.append(ll)
            return imgs, lbls
        else:
            return apply_one(img, lbl)

    def __call__(self, img, lbl):
        img = self.img_transforms(img)
        lbl = self.lbl_transforms(lbl)
        
        img, lbl = self.resize(img, lbl)
        img, lbl = self.geometric_transforms(img, lbl)

        if isinstance(lbl, list):
            lbl = [l for l in lbl]
        else:
            lbl = lbl
        return img, lbl


class MapTransformTest:
    def __init__(self, image_size=384, color_jitter=0.01):
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

        # self.pad_transforms = v2.Compose([
        #     PadToTargetSize(image_size),
        # ])

        self.resize_img = v2.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)
        self.resize_lbl = v2.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST)
    
    def resize(self, imgs, lbls):
        """
        Resize images and labels to target size. Supports either a single image/label
        or lists of them.
        Images use bilinear interpolation and labels use nearest.
        """
        def apply_one(i, l):
            i = self.resize_img(i)
            if l is not None:
                l = self.resize_lbl(l.unsqueeze(0)).squeeze(0)
                try:
                    l = l.to(torch.bool)
                except Exception:
                    # fallback if returned type isn't a tensor
                    pass
            return i, l

        if isinstance(imgs, list):
            img_list = []
            lbl_list = []
            for i, ll in zip(imgs, lbls):
                ii, ll = apply_one(i, ll)
                img_list.append(ii)
                lbl_list.append(ll)
            return img_list, lbl_list
        else:
            return apply_one(imgs, lbls)

    def __call__(self, img, lbl=None):
        img = self.img_transforms(img)
        if lbl is not None:
            lbl = self.lbl_transforms(lbl)
            img, lbl = self.resize(img, lbl)
            return img, lbl
        else:
            img, lbl = self.resize(img, lbl)
            return img


if __name__ == '__main__':
    dataset = SMOLDataset("/koko/datasets/SMOL_syntra", info_file="train_100shot.txt", mode="train")
    # dataset = SynTraDataset("/koko/datasets/SMOL_syntra", info_file="pairing_train_100shot_train_100shot_intercls.json", mode="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)
    for d in dataset:
         src_img, src_lbl, tgt_img, tgt_lbl, aux = d
         if src_lbl.sum() == 0 and tgt_lbl.sum() > 0:
             print(src_img.shape, src_lbl.shape, tgt_img.shape, tgt_lbl.shape, aux)
             print("empty src lbl")
