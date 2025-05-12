import numpy as np
import torch
import cv2


class Evaluator:
    def __init__(self):
        self.tp = 0
        self.union = 0
        self.pos_pred = 0
        self.pos_gt = 0
        self.overall_tp = 0 # including background
        self.n = 0

    def add_sample(self, pred, target, thresh=0.5):
        assert pred.shape == target.shape
        pred_cls = pred > thresh

        # iou
        tp = torch.logical_and(pred_cls, target)
        union = torch.logical_or(pred_cls, target)

        self.n += 1
        self.overall_tp += (pred_cls == target).sum() / target.numel()
        self.tp += torch.sum(tp)
        self.union += torch.sum(union)
        self.pos_pred += torch.sum(pred_cls)
        self.pos_gt += torch.sum(target)

    def eval(self):
        iou = self.tp / self.union
        prec = self.tp / self.pos_pred
        rec = self.tp / self.pos_gt
        acc = self.overall_tp / self.n

        return iou, prec, rec, acc

def cal_all(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection.sum() / union.sum()
    pos_pred = pred_mask.sum()
    pos_gt = gt_mask.sum()
    prec = intersection / pos_pred
    rec = intersection / pos_gt
    return iou, prec, rec


def eval_map_sheet():

    sheet = "3922_1974"
    path_pred_unet = f"/home/yuan/data/out/LGLN/lgln-1973-1975/epoch20/holistic_seg_result/{sheet}.png"
    path_gt = f"/home/yuan/data/HisMap/LGLN/dataset/test/label/{sheet}.png"

    pred_unet = cv2.imread(path_pred_unet, cv2.IMREAD_GRAYSCALE)
    pred_unet = pred_unet / 50
    pred_unet[pred_unet == 5] = 4
    gt = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE) / 50

    for label_idx in range(1, 5):
        path_pred = f"/home/yuan/code/HisMap/smolsam/logs/cosine_lr5e-5/map_pred/{sheet}_mask{label_idx}.png"
        pred = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)
        pred_mask = pred > 0.5
        gt_mask = gt == label_idx

        h = min(pred_mask.shape[0], gt_mask.shape[0])
        w = min(pred_mask.shape[1], gt_mask.shape[1])
        pred_mask = pred_mask[:h, :w]
        gt_mask = gt_mask[:h, :w]
        pred_unet_i = pred_unet == label_idx
        pred_unet_i = pred_unet_i[:h, :w]

        print("LABEL:", label_idx)
        iou, prec, rec = cal_all(pred_mask, gt_mask)
        print("SMOL:",  iou, prec, rec)
        iou, prec, rec = cal_all(pred_unet_i, gt_mask)
        print("UNET:",  iou, prec, rec)


eval_map_sheet()