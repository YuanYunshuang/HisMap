import os

import numpy as np
import tqdm
from geotiff import GeoTiff
import cv2


def read_tiff(filename):
    if not os.path.exists(filename):
        print('File not found: {}'.format(filename))
        return None
    geo_tiff = GeoTiff(filename)
    tiff_array = np.array(geo_tiff.read())
    return tiff_array


def resize_tiff(filename, ratio=1.0):
    img = read_tiff(filename)
    img = cv2.resize(img, None, interpolation=cv2.INTER_AREA, fx=ratio, fy=ratio)
    return img


def resize_tiffs(dirname, ratio=1.0):
    output_dir = dirname + f'_dx_{int(1 /ratio)}'
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(dirname):
        print(filename)
        img = resize_tiff(os.path.join(dirname, filename), ratio=ratio)
        cv2.imwrite(os.path.join(output_dir, filename.replace('tif', 'png')), img)


def norm_img(img):
    img = (img - img.min()) / np.clip(
        img.max() - img.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)
    return img


def generate_train_data(root_dir, size=1024, margin=512):
    gt_dir = os.path.join(root_dir, 'labels_man')
    img_dir = os.path.join(root_dir, 'tifs')

    gt_files = os.listdir(gt_dir)

    output_dir = os.path.join(root_dir, 'train')
    os.makedirs(os.path.join(output_dir, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'gts'), exist_ok=True)

    for f in tqdm.tqdm(gt_files):
        name = f.split('.')[0]
        img_file = os.path.join(img_dir, name + '.tif')
        if not os.path.exists(img_file):
            continue

        img = read_tiff(img_file)
        img = norm_img(img)
        if img.ndim == 2:
            img = img[..., None].repeat(3, axis=-1)
        label = cv2.imread(os.path.join(gt_dir, f), cv2.IMREAD_GRAYSCALE) // 50
        H, W = min(label.shape[0], img.shape[0]), min(label.shape[1], img.shape[1])

        xs = np.arange(0, H, margin)
        ys = np.arange(0, W, margin)
        xs[-1] = H - size
        ys[-1] = W - size
        xys = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        for xy in xys:
            x, y = xy
            img_crop = img[x:x + size, y:y + size]
            label_crop = label[x:x + size, y:y + size]
            if np.sum(label_crop > 0) < 100:
                continue
            unique_ids = np.unique(label_crop)[1:]
            if len(unique_ids) == 0:
                continue
            np.save(os.path.join(output_dir, 'imgs', f'{name}_{x}_{y}.npy'), img_crop)
            np.save(os.path.join(output_dir, 'gts', f'{name}_{x}_{y}.npy'), label_crop)
            # cv2.imwrite(os.path.join(output_dir, 'imgs', f'{name}_{x}_{y}.png'), img_crop)
            # cv2.imwrite(os.path.join(output_dir, 'gts', f'{name}_{x}_{y}.png'), label_crop)


def remove_empty(path):
    gt_files = os.listdir(path)
    for f in gt_files:
        data = np.load(
            os.path.join(path, f), "r", allow_pickle=True
        )
        img = np.load(
            os.path.join(path.replace('gts', 'imgs'), f), "r", allow_pickle=True
        )
        ids = np.unique(data)[1:]
        if len(ids) == 0:
            os.remove(os.path.join(path, f))
            cv2.imshow('tmp', img)
            cv2.waitKey()
            cv2.destroyWindow('tmp')
            print(f)
            print(ids)


if __name__ == '__main__':
    # resize_tiffs('/home/yuan/data/HisMap/MapSAM/tifs', ratio=0.1)
    remove_empty('/home/yuan/data/HisMap/MapSAM/train/gts')
    # generate_train_data('/home/yuan/data/HisMap/MapSAM')