import glob
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
    for f in tqdm.tqdm(gt_files):
        data = np.load(
            os.path.join(path, f), "r", allow_pickle=True
        )
        img = np.load(
            os.path.join(path.replace('gts', 'imgs'), f), "r", allow_pickle=True
        )
        ids = np.unique(data).tolist()
        if 3 not in ids:
            os.remove(os.path.join(path, f))
        elif img.shape[0] != 1024 or img.shape[1] != 1024 or data.shape[0] != 1024 or data.shape[1] != 1024:
            os.remove(os.path.join(path, f))
            # cv2.imshow('tmp', img)
            # cv2.waitKey()
            # cv2.destroyWindow('tmp')
            # print(f)
            # print(np.unique(data))


def region_growing(img_file, threshold=1):
    import matplotlib.pyplot as plt
    # Load the image
    img = np.load(img_file, "r", allow_pickle=True)
    visited = img == 0
    region = np.zeros_like(img, dtype=int)

    nbrs = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    inds = np.where(np.logical_not(visited))
    seed = np.stack(inds, axis=-1)[:1]
    visited[seed[0, 0], seed[0, 1]] = True
    seed_value = img[seed[0, 0], seed[0, 1]]
    oid = 1
    region[seed[0, 0], seed[0, 1]] = oid
    itr = 0

    while len(seed) > 0:
        cur_nbrs = seed[:, None] + nbrs[None, :]
        cur_nbrs = np.unique(cur_nbrs.reshape(-1, 2), axis=0)
        mask = (0 <= cur_nbrs[:, 0]) & (cur_nbrs[:, 0] < img.shape[0]) & \
                     (0 <= cur_nbrs[:, 1]) & (cur_nbrs[:, 1] < img.shape[1])
        cur_nbrs = cur_nbrs[mask]
        valid_mask = (img[cur_nbrs[:, 0], cur_nbrs[:, 1]] - seed_value < threshold) & \
                     (np.logical_not(visited[cur_nbrs[:, 0], cur_nbrs[:, 1]]))

        valid_nbrs = cur_nbrs[valid_mask]
        visited[valid_nbrs[:, 0], valid_nbrs[:, 1]] = True
        region[valid_nbrs[:, 0], valid_nbrs[:, 1]] = oid
        seed = valid_nbrs

        if len(seed) == 0 and not visited.all():
            oid += 1
            inds = np.where(np.logical_not(visited))
            seed = np.stack(inds, axis=-1)[:1]
            region[seed[0, 0], seed[0, 1]] = oid
            visited[seed[0, 0], seed[0, 1]] = True
            seed_value = img[seed[0, 0], seed[0, 1]]

        itr += 1

    # seg = np.zeros_like((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # for i in range(oid):
    #     seg = seg + (region == (i + 1))[:, :, None] * np.random.randint(256, size=(1, 1, 3))
    # # Display the result
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title('Original Image')
    # plt.imshow(img * 50, cmap='gray')
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.title('Segmented Region')
    # plt.imshow(seg, cmap='gray')
    # plt.axis('off')
    #
    # plt.savefig("/home/yuan/Downloads/tmp.png")
    # plt.close()
    return region.astype(np.uint8)


def npy_to_img(in_path, out_path):
    for f in os.listdir(in_path):
        name = os.path.join(in_path, f)
        img = np.load(name, "r", allow_pickle=True)
        img = (np.array(img) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_path, f.replace('npy', 'jpg')), img)


if __name__ == '__main__':
    # resize_tiffs('/home/yuan/data/HisMap/MapSAM/tifs', ratio=0.1)
    # remove_empty('/home/yuan/data/HisMap/MapSAM/train/gts')
    # generate_train_data('/home/yuan/data/HisMap/MapSAM')
    # for f in tqdm.tqdm(glob.glob('/home/yuan/data/HisMap/MapSAM/train/gts/*.npy')):
    #     region = region_growing(f)
    #     np.save(os.path.join(f), region)
    npy_to_img(
        "/home/yuan/data/HisMap/MapSAM/train/imgs",
        "/home/yuan/data/HisMap/MapSAM/train/imgs_vis"
    )