import glob
import os
import shutil

import numpy as np
import cv2
from src.tools.map_analyzer import read_tiff


def evaluate(pred, gt):
    pred = pred[:gt.shape[0], :gt.shape[1]]
    pred = pred / 255 > 0.5

    union = np.logical_or(pred, gt)
    intersection = np.logical_and(pred, gt)
    iou = np.sum(intersection) / np.sum(union)
    print(iou)


def draw_pred_gt_img(pred_img, gt):
    img = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    img[..., 1] = gt * 255
    img[..., 0] = pred_img[:gt.shape[0], :gt.shape[1]]
    cv2.imwrite("/media/yuan/luna/data/HisMAP/Hameln/tmp.png", img)


def copy_pseudo_labels():
    path_in = "/media/yuan/luna/data/HisMAP/Hameln/exp_soft"
    path_out = "/media/yuan/luna/data/HisMAP/Hameln/for_annotation"
    dirs = os.listdir(path_in)
    for d in dirs:
        files = glob.glob(os.path.join(path_in, d, "epoch10/holistic_seg_result/*.png"))
        for f in files:
            shutil.copy(f, os.path.join(path_out, os.path.basename(f)))


if __name__ == '__main__':
    copy_pseudo_labels()
    cls = 'set'
    gt = read_tiff(f'/media/yuan/luna/data/HisMAP/Hameln/gt_1897/3921_1897_{cls}.tif')

    pred = cv2.imread(f"/media/yuan/luna/data/HisMAP/Hameln/exp_soft/seg4cls_1914_1919/epoch10/holistic_seg_result/3921_1897_{cls}.png",
                          cv2.IMREAD_UNCHANGED)
    draw_pred_gt_img(pred, gt)
    evaluate(pred, gt)

    # pred_all = cv2.imread(
    #     f"/media/yuan/luna/data/HisMAP/Hameln/exp_0.5/seg4cls_all/epoch50/holistic_seg_result/3921_1897_{cls}.png",
    #     cv2.IMREAD_UNCHANGED)
    # # draw_pred_gt_img(pred_all, gt)
    # evaluate(pred_all, gt)
