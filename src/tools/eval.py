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


if __name__ == '__main__':
    # for exp in ['seg4cls_edl', 'seg4cls_focal', 'seg4cls_ce', 'seg4cls_focedl']:
    #     print(exp)
    #     eval(f'/home/yuan/code/HisMap/logs/{exp}/epoch50/holistic_seg_result',
    #          '/home/yuan/data/HisMap/Hameln/raster_gt',
    #          'for')

    # draw_contra_map(
    #     f'/home/yuan/code/HisMap/logs/seg4cls_ce/epoch50/holistic_seg_result',
    #      '/home/yuan/data/HisMap/Hameln/raster_gt',
    #      'for')

    map_dict = {}
    H = W = 20000000
    for mp in ['3821_2016', '3822_2016', '3921_2017', '3922_2017']:
        tif = read_tiff(f"/home/yuan/data/HisMap/Hameln/TK_projektiv_entzerrt_1m/{mp}.tif")
        map_dict[mp] = tif[::5, ::5, ::-1]
        H = min(H, map_dict[mp].shape[0])
        W = min(W, map_dict[mp].shape[1])
    full_map1 = np.concatenate([map_dict['3821_2016'][:H, :W], map_dict['3822_2016'][:H, :W]], axis=1)
    full_map2 = np.concatenate([map_dict['3921_2017'][:H, :W], map_dict['3922_2017'][:H, :W]], axis=1)
    full_map = np.concatenate([full_map1, full_map2], axis=0)
    cv2.imwrite("/home/yuan/Downloads/fullmap.jpg", full_map)


