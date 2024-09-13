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


if __name__ == '__main__':
    cls = 'for'
    gt = read_tiff(f'/media/yuan/luna/data/HisMAP/Hameln/gt_1897/3921_1897_{cls}.tif')

    pred_05 = cv2.imread(f"/media/yuan/luna/data/HisMAP/Hameln/exp_0.5/seg4cls_1914_1919/epoch10/holistic_seg_result/3921_1897_{cls}.png",
                          cv2.IMREAD_UNCHANGED)
    # draw_pred_gt_img(pred_05, gt)
    evaluate(pred_05, gt)

    pred_all = cv2.imread(
        f"/media/yuan/luna/data/HisMAP/Hameln/exp_0.5/seg4cls_all/epoch50/holistic_seg_result/3921_1897_{cls}.png",
        cv2.IMREAD_UNCHANGED)
    # draw_pred_gt_img(pred_all, gt)
    evaluate(pred_all, gt)
