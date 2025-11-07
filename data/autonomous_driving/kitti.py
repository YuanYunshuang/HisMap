import os
import numpy as np
from PIL import Image

color_map = {
    'dynamic': (111, 74, 0),
    'ground': (81, 0, 81),
    'road': (128, 64, 128),
    'sidewalk': (244, 35, 232),
    'parking': (250, 170, 160),
    'rail track': (230, 150, 140),
    'building': (70, 70, 70),
    'wall': (102, 102, 156),
    'fence': (190, 153, 153),
    'guard rail': (180, 165, 180),
    'bridge': (150, 100, 100),
    'tunnel': (150, 120, 90),
    'pole': (153, 153, 153),
    'polegroup': (153, 153, 153),
    'traffic light': (250, 170, 30),
    'traffic sign': (220, 220, 0),
    'vegetation': (107, 142, 35),
    'terrain': (152, 251, 152),
    'sky': (70, 130, 180),
    'person': (220, 20, 60),
    'rider': (255, 0, 0),
    'car': (0, 0, 142),
    'truck': (0, 0, 70),
    'bus': (0, 60, 100),
    'caravan': (0, 0, 90),
    'trailer': (0, 0, 110),
    'train': (0, 80, 100),
    'motorcycle': (0, 0, 230),
    'bicycle': (119, 11, 32),
    'license plate': (0, 0, 142),
}

label_id_map = {
    'dynamic': 5,
    'ground': 6,
    'road': 7,
    'sidewalk': 8,
    'parking': 9,
    'rail track': 10,
    'building': 11,
    'wall': 12,
    'fence': 13,
    'guard rail': 14,
    'bridge': 15,
    'tunnel': 16,
    'pole': 17,
    'polegroup': 18,
    'traffic light': 19,
    'traffic sign': 20,
    'vegetation': 21,
    'terrain': 22,
    'sky': 23,
    'person': 24,
    'rider': 25,
    'car': 26,
    'truck': 27,
    'bus': 28,
    'caravan': 29,
    'trailer': 30,
    'train': 31,
    'motorcycle': 32,
    'bicycle': 33,
    'license plate': -1,
}


kitti_color_palette = [0, 0, 0]  # background color
for class_name in color_map:
    kitti_color_palette.extend(color_map[class_name])
kitti_color_palette.extend([0,] * (255*3 - len(kitti_color_palette)))  # pad the palette to 256 colors
kitti_color_palette.extend([255, 255, 255])  # nonlabeled color


def convert(input_path, output_path):
