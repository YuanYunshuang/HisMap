import os

import cv2
import numpy as np


def generate_smolsam_data(indir, img_size=384, margin=0):
    outdir = indir.replace('processed', 'smolsam')
    os.makedirs(os.path.join(outdir, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'lbls'), exist_ok=True)

    mapfiles = os.listdir(os.path.join(indir, 'maps'))

    for mf in mapfiles:
        _, sheet, _, year = mf.split('.')[0].split('_')[:4]
        name = f"{sheet}_{year}"
        MAP = cv2.imread(os.path.join(indir, 'maps', mf))
        LABEL = cv2.imread(os.path.join(indir, 'lbls', mf), cv2.IMREAD_GRAYSCALE)

        H, W = min(MAP.shape[0], LABEL.shape[0]), min(MAP.shape[1], LABEL.shape[1])
        x = np.arange(0, W, img_size-margin)
        y = np.arange(0, H, img_size-margin)
        x[-1] = W - img_size
        y[-1] = H - img_size
        yy, xx = np.meshgrid(x, y)
        xys = np.vstack((xx.flatten(), yy.flatten())).T

        for xy in xys:
            img = MAP[xy[0]:xy[0] + img_size, xy[1]:xy[1] + img_size]
            lbl = LABEL[xy[0]:xy[0] + img_size, xy[1]:xy[1] + img_size]
            if img.ndim == 2:
                img = img[..., np.newaxis].repeat(3, axis=2)
            img = np.pad(img, ((0, img_size - img.shape[0]), (0, img_size - img.shape[1]), (0, 0)),
                         'constant', constant_values=0)
            lbl = np.pad(lbl, ((0, img_size - lbl.shape[0]), (0, img_size - lbl.shape[1])),
                         'constant', constant_values=255)

            cv2.imwrite(os.path.join(outdir, 'imgs', f'{name}_{xy[0]}_{xy[1]}.png'), img)
            cv2.imwrite(os.path.join(outdir, 'lbls', f'{name}_{xy[0]}_{xy[1]}.png'), lbl)

if __name__ == '__main__':
    generate_smolsam_data("/home/yuan/data/HisMap/Bayern/processed")