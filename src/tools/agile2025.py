import glob
import math
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

from src.tools.map_analyzer import read_tiff
from src.utils.misc import load_json, save_json

def draw_attn_scale():

    # Create a gradient image
    gradient = np.linspace(0, 1, 256).reshape(1, -1)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 1))

    # Display the colormap with alpha=0.5
    ax.imshow(gradient, aspect='auto', cmap='jet', alpha=0.5)

    # Remove axes for clarity
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    # Show the plot
    plt.savefig("/home/yuan/Downloads/attn_scale.pdf")
    plt.close()


def stich_attn_maps(path, x, y, step=384, n_step=5):
    """
        stich_attn_maps("/home/yuan/data/out/hismap_attn/settlement_p64/imgs", 0, 384)
    """
    images = []
    for i in range(n_step):
        imgs = []
        for j in range(n_step):
            x_ = x + i * step
            y_ = y + j * step
            filename = glob.glob(os.path.join(path, f'3922_1974_{x_}_{y_}_*.jpg'))[0]
            img = cv2.imread(filename)[10:-10, 10:-10]
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=0)
        images.append(imgs)
    images = np.concatenate(images, axis=1)
    cv2.imwrite(f"{path}.jpg", images)


def cal_iou(path, label_file, fg_label=0, cls_name=''):
    """
        cal_iou(
        "/home/yuan/data/out/hismap_attn/wood_ep100",
        "/home/yuan/data/HisMap/LGLN/dataset/age_tracing_data/test/label/3922_1974.png",
        fg_label=1, cls_name='wood'
    )
    """
    label = cv2.imread(label_file, cv2.IMREAD_UNCHANGED) // 50
    label = label == fg_label
    H, W = label.shape[:2]
    pred = np.zeros((H, W))
    M = np.zeros((H, W, 3), np.uint8)
    img_size = 384
    for h in range(0, H, img_size):
        for w in range(0, W, img_size):
            json_file = os.path.join(path, 'inference', f"3922_1974_{w}_{h}.json")
            img_file = os.path.join(path, 'imgs', f"3922_1974_{w}_{h}_*.jpg")

            data = load_json(json_file)
            img = cv2.imread(glob.glob(img_file)[0], cv2.IMREAD_UNCHANGED)[8:-8, 8:-8]

            attn_ = np.array(data['attn'])
            attn = np.zeros_like(attn_)
            for i, idx in enumerate(data['idx']):
                attn[idx] = attn_[i]
            attn_size = int(math.sqrt(attn.shape[0]))
            attn = attn.reshape(attn_size, attn_size)
            s = img_size // attn_size
            attention_map_resized = np.kron(attn, np.ones((s, s)))
            pred_mask = attention_map_resized
            try:
                h2 = min(h+img_size, H)
                w2 = min(w+img_size, W)
                pred[h:h2, w:w2] = pred_mask[:h2 - h, :w2 - w]
                M[h:h2, w:w2] = img[:h2 - h, :w2 - w]
            except Exception as e:
                print(h, w, h2, w2)
                raise e
            # pred[w:min(w+img_size, W), h:min(h+img_size, H)] = pred_mask[:min(w+img_size, W) - w, :min(h+img_size, H) - h]
            # M[w:min(w+img_size, W), h:min(h+img_size, H)] = img[:min(w+img_size, W) - w, :min(h+img_size, H) - h]

    cv2.imwrite(f"{path}/res_{cls_name}_mask.jpg", (pred * 255).astype(np.uint8))
    cv2.imwrite(f"{path}/res_{cls_name}_label.jpg", label.astype(np.uint8) * 255)
    cv2.imwrite(f"{path}/res_{cls_name}_img.jpg", M)

    s = 11188
    label = label[:s, :s]
    pred = pred[:s, :s]
    res = []
    for thr in range(0, 10):
        pred_mask = pred > (thr * 0.1)
        tp = np.logical_and(label, pred_mask).sum()
        fp = np.logical_and(np.logical_not(label), pred_mask).sum()
        fn = np.logical_and(np.logical_not(pred_mask), label).sum()
        union = np.logical_or(label, pred_mask).sum()
        iou = tp / union
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        res.append([thr * 0.1, iou, precision, recall])
        print("Thr | IoU | Prec | Rec:", f"{thr * 0.1:.1f} | {iou:.4f} | {precision:.4f} | {recall:.4f}")
    save_json(res, f"{path}/res_{cls_name}.json")

def plot_iou_prec(path_dict, out_file):
    """
        plot_iou_prec(
        {
            "wood": "/home/yuan/data/out/hismap_attn/wood_ep100/res_wood.json",
            "settlement": "/home/yuan/data/out/hismap_attn/settlement_p64/res_settlement.json"
        },
        ""/home/yuan/Downloads/iou_prec.pdf"
        )
    """
    fig = plt.figure(figsize=(6, 3))
    axs = fig.subplots(1, len(path_dict))

    i = 0
    for k, v in path_dict.items():
        ax = axs[i]
        data = np.array(load_json(v))
        ax.plot(data[:, 0], data[:, 1], '-o', label='IoU')
        ax.plot(data[:, 0], data[:, 2], '-*', label='Precision')
        ax.plot(data[:, 0], data[:, 3], '-^', label='Recall')
        ax.set_title(k)
        ax.legend()
        ax.set_xlabel('Threshold')
        ax.set_ylabel('IoU & Precision & Recall')
        i += 1
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def rezize_img(path, ratio=0.2):
    if 'dx' in path:
        return
    ext = path.split('.')[-1]
    if 'tif' == ext:
        img = read_tiff(path)[..., ::-1]
    else:
        img = cv2.imread(path)

    img = cv2.resize(img, None, interpolation=cv2.INTER_AREA, fx=ratio, fy=ratio)
    cv2.imwrite(path.replace(f'.{ext}', f'_dx{int(1/ratio)}.jpg'), img)


def crop_quadrant(path, quadrant=1):
    ext = path.split('.')[-1]
    if 'tif' == ext:
        img = read_tiff(path)[..., ::-1]
    else:
        img = cv2.imread(path)
    if quadrant == 2:
        img = img[:img.shape[0] // 2, :img.shape[1] // 2]
    elif quadrant == 1:
        img = img[:img.shape[0] // 2, img.shape[1] // 2:]
    elif quadrant == 3:
        img = img[img.shape[0] // 2:, :img.shape[1] // 2]
    elif quadrant == 4:
        img = img[img.shape[0] // 2:, img.shape[1] // 2:]
    cv2.imwrite(path.replace(f'.{ext}', f'_{quadrant}.jpg'), img)

if __name__ == '__main__':
    pass
    # cal_iou(
    #     "/home/yuan/data/out/hismap_attn/settlement_p64",
    #     "/home/yuan/data/HisMap/LGLN/dataset/age_tracing_data/test/label/3922_1974.png",
    #     fg_label=3, cls_name='settlement'
    # )

    # cal_iou(
    #     "/home/yuan/data/out/hismap_attn/wood_ep100",
    #     "/home/yuan/data/HisMap/LGLN/dataset/age_tracing_data/test/label/3922_1974.png",
    #     fg_label=1, cls_name='wood'
    # )

    # plot_iou_prec(
    #     {
    #         "Wood": "/home/yuan/data/out/hismap_attn/wood_ep100/res_wood.json",
    #         "Settlement": "/home/yuan/data/out/hismap_attn/settlement_p64/res_settlement.json"
    #     },
    #     "/home/yuan/Downloads/iou_prec.pdf"
    # )

    rezize_img("/home/yuan/data/HisMap/LGLN/dataset/3821_1974.tif", ratio=0.4)
    # crop_quadrant("/home/yuan/data/HisMap/LGLN/dataset/3821_1974_0.jpg", quadrant=1)