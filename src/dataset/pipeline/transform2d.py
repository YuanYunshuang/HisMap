import cv2
import torch
from torchvision.transforms import v2
from PIL import Image


class FlipRotImage:
    def __init__(self, degrees=180):
        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=(-degrees, degrees)),
        ])

    def __call__(self, data_dict):
        img_keys = [k for k in data_dict.keys() if 'img' in k]
        imgs = [data_dict[k] for k in img_keys]
        imgs = self.transforms(imgs)
        for i, k in enumerate(img_keys):
            data_dict[k] = imgs[i]
        return data_dict


class FormatImage:
    def __init__(self,):
        self.transforms_input = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.transforms_gt = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=False),
        ])

    def __call__(self, data_dict):
        for k, v in data_dict.items():
            if 'img' in k:
                if 'input' in k:
                    data_dict[k] = self.transforms_input(v)
                elif 'gt' in k:
                    data_dict[k] = self.transforms_gt(v)
                else:
                    raise ValueError(f'Unexpected key {k}')
        # img_keys = [k for k in data_dict.keys() if 'img' in k]
        # imgs = [data_dict[k] for k in img_keys]
        # imgs = self.transforms(imgs)
        # for i, k in enumerate(img_keys):
        #     data_dict[k] = imgs[i]
        return data_dict