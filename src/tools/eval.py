import os

import cv2
import numpy as np
from glob import glob

from src.tools.map_analyzer import read_tiff


def iou_and_acc(pred, gt):
    intersection = np.logical_and(pred, gt)
    area_intersection = np.sum(intersection)
    area_union = np.sum(np.logical_or(pred, gt))
    area_gt = np.sum(gt > 0)
    iou = area_intersection / area_union
    acc = area_intersection / area_gt

    return iou, acc


def eval(res_dir, gt_dir, cls='for'):
    res_files = glob(os.path.join(res_dir, f'*{cls}.png'))
    outstream = ""
    for rf in res_files:
        basename = os.path.basename(rf).split('.')[0]
        name = basename + '_r.png'
        res = cv2.imread(rf, cv2.IMREAD_UNCHANGED) / 255 > 0.5
        gt = cv2.imread(os.path.join(gt_dir, name))[..., 0]
        H = min(res.shape[0], gt.shape[0])
        W = min(res.shape[1], gt.shape[1])
        iou, acc = iou_and_acc(res[:H, :W], gt[:H, :W])
        ss = f"{basename}: IoU= {iou*100:.01f}, Acc= {acc*100:.01f}"
        print(ss)
        outstream += ss + '\n'

    with open(os.path.join(os.path.dirname(res_dir), 'eval.txt'), 'w') as f:
        f.write(outstream)


def draw_contra_map(res_dir, gt_dir, cls='for'):
    res_files = glob(os.path.join(res_dir, f'*{cls}.png'))
    for rf in res_files:
        basename = os.path.basename(rf).split('.')[0]
        name = basename + '_r.png'
        res = cv2.imread(rf, cv2.IMREAD_UNCHANGED) / 255 > 0.5
        gt = cv2.imread(os.path.join(gt_dir, name), cv2.IMREAD_UNCHANGED)
        H = min(res.shape[0], gt.shape[0])
        W = min(res.shape[1], gt.shape[1])
        gt[:H, :W, 0] = (res[:H, :W] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(os.path.dirname(res_dir), 'contra_map', basename + '.png'), gt[:H, :W])


def eval_lgln(res_dir, gt_dir):
    maps = os.listdir(res_dir)
    eval_res = []
    ss = ""
    for m in maps:
        pred_file = os.path.join(res_dir, m)
        label_file = os.path.join(gt_dir, m)
        pred = cv2.imread(pred_file, cv2.IMREAD_UNCHANGED) / 50
        gt = cv2.imread(label_file, cv2.IMREAD_UNCHANGED) / 50
        pred = pred[:gt.shape[0], :gt.shape[1]]
        ious = []
        ss += f"{m.split('.')[0]}:\n"
        for i in range(1, 6):
            pred_mask = pred == i
            gt_mask = gt == i
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            iou = intersection / max(union, 1)
            ss += f"  {iou*100:.1f}"
            ious.append(iou)
        acc = (pred == gt).sum() / gt.size
        ss += f"  |  {acc*100:.1f}\n"
        eval_res.append(ious)
    print(ss)
    with open(os.path.join(os.path.dirname(res_dir), 'eval.txt'), 'w') as f:
        f.write(ss)



if __name__ == '__main__':
    eval_lgln(
        "/home/yuan/data/out/LGLN/pseudo-0.5/pseudo_12-04-23-14-49/epoch5/holistic_seg_result",
        "/home/yuan/data/HisMap/LGLN/dataset/test/label"
    )


