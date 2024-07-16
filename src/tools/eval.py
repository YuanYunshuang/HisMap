import cv2
import numpy as np
from glob import glob


def iou_and_acc(pred, gt):
    intersection = np.logical_and(pred, gt)
    area_intersection = np.sum(intersection)
    area_union = np.sum(np.logical_or(pred, gt))
    area_gt = np.sum(gt > 0)
    iou = area_intersection / area_union
    acc = area_intersection / area_gt

    return iou, acc


def eval(res_dir, gt_dir):
    res_files = glob(res_dir + '*.png')
