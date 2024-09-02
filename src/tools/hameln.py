import os
from glob import glob

import cv2
import numpy as np

from src.tools.map_analyzer import read_tiff

MAP_IDS = [3922, 3921, 3822, 3821]
YEARS = [2017, 2016, 2012, 2011, 2005, 2002, 1996, 1992, 1986, 1982, 1978, 1974, 1960, 1955, 1952, 1945, 1936, 1925, 1898]
CATEGORY = ['for']


def generate_dataset(map_dir, raster_dir, out_dir, year, category, img_size=512, overlap=24):
    out_dir_sample = os.path.join(out_dir, 'sample')
    out_dir_gt = os.path.join(out_dir, 'gt')
    os.makedirs(out_dir_sample, exist_ok=True)
    os.makedirs(out_dir_gt, exist_ok=True)
    for id in MAP_IDS:
        print(id)
        mapfiles = glob(os.path.join(map_dir, f'{id}_*.tif'))
        years = np.array([int(os.path.basename(x).split('.')[0].split('_')[1]) for x in mapfiles])
        idx = np.argmin(np.abs(years - year))
        nearest_year = years[idx]
        map_array = read_tiff(mapfiles[idx])
        rasterfile = os.path.join(raster_dir, f'{id}_2023_{category}.tif')
        raster_array = read_tiff(rasterfile)

        H, W = map_array.shape[:2]

        for h in range(0, H, img_size - overlap):
            h_end = min(h + img_size, H)
            for w in range(0, W, img_size - overlap):
                w_end = min(w + img_size, W)
                sample = map_array[h:h_end, w:w_end]
                gt = raster_array[h:h_end, w:w_end]

                if sample.shape[:2] != (img_size, img_size):
                    padh = img_size - sample.shape[0]
                    padw = img_size - sample.shape[1]
                    sample = np.pad(sample, ((0, padh), (0, padw), (0, 0)), mode='constant', constant_values=0)
                    gt = np.pad(gt, ((0, padh), (0, padw)), mode='constant', constant_values=255)
                filename = f"{id}_{nearest_year}_{h}_{w}.png"
                cv2.imwrite(os.path.join(out_dir_sample, filename), sample.astype(np.uint8))
                cv2.imwrite(os.path.join(out_dir_gt, filename), gt.astype(np.uint8))


def generate_4cls_dataset(map_dir, raster_dir, out_dir, categories, year=None, img_size=512, overlap=24):
    out_dir_sample = os.path.join(out_dir, 'sample')
    out_dir_gt = os.path.join(out_dir, 'gt')
    os.makedirs(out_dir_sample, exist_ok=True)
    os.makedirs(out_dir_gt, exist_ok=True)
    for id in MAP_IDS:
        print(id)
        mapfiles = glob(os.path.join(map_dir, f'{id}_*.tif'))
        years = np.array([int(os.path.basename(x).split('.')[0].split('_')[1]) for x in mapfiles])
        if year is not None:
            idx = np.argmin(np.abs(years - year))
            years = [years[idx]]

        # get initial training labels from raster labels of 2023
        H = W = None
        raster_arr = None
        for i, category in enumerate(categories):
            rasterfile = os.path.join(raster_dir, f'{id}_2023_{category}.tif')
            raster = read_tiff(rasterfile)
            if raster_arr is None:
                raster_arr = np.zeros_like(raster, dtype=np.uint8)
                H, W = raster_arr.shape[:2]
            inds = np.where(raster < 255)
            raster_arr[inds[0], inds[1]] = i + 1

        for i, cur_year in enumerate(years):
            map_array = read_tiff(os.path.join(map_dir, f'{id}_{cur_year}.tif'))
            if map_array.ndim == 2:
                map_array = map_array[:, :, np.newaxis].repeat(3, axis=2)
            for h in range(0, H, img_size - overlap):
                h_end = min(h + img_size, H)
                for w in range(0, W, img_size - overlap):
                    w_end = min(w + img_size, W)
                    sample = map_array[h:h_end, w:w_end]
                    if i == 0:
                        gt = raster_arr[h:h_end, w:w_end]

                    if sample.shape[:2] != (img_size, img_size):
                        padh = img_size - sample.shape[0]
                        padw = img_size - sample.shape[1]
                        try:
                            sample = np.pad(sample, ((0, padh), (0, padw), (0, 0)), mode='constant', constant_values=0)
                        except:
                            print('d')
                        if i == 0:
                            gt = np.pad(gt, ((0, padh), (0, padw)), mode='constant', constant_values=0)
                    cv2.imwrite(os.path.join(out_dir_sample, f"{id}_{cur_year}_{h}_{w}.png"), sample.astype(np.uint8))
                    if i == 0:
                        cv2.imwrite(os.path.join(out_dir_gt, f"{id}_{h}_{w}.png"), gt.astype(np.uint8))


def generate_4cls_pseudo_labels(pred_dir, out_dir_gt, categories, img_size=512, overlap=24):
    # out_dir_gt = os.path.join(out_dir, 'gt_pseudo')
    os.makedirs(out_dir_gt, exist_ok=True)
    patches = [x for x in os.listdir(pred_dir) if categories[0] in x]
    for patch in patches:
        id, year = os.path.splitext(patch)[0].split('_')[:2]
        print(id, year)
        H = W = None
        raster_arr = None
        for i, category in enumerate(categories):
            rasterfile = os.path.join(pred_dir, f'{id}_{year}_{category}.png')
            raster = cv2.imread(rasterfile, cv2.IMREAD_UNCHANGED)
            raster = raster > 255 * 0.3
            # cv2.imwrite(os.path.join(pred_dir, f'{id}_{year}_{category}_tmp.png'), raster_tmp.astype(np.uint8))
            if raster_arr is None:
                raster_arr = np.zeros_like(raster, dtype=np.uint8)
                H, W = raster_arr.shape[:2]
            inds = np.where(raster)
            raster_arr[inds[0], inds[1]] = i + 1

        for h in range(0, H, img_size - overlap):
            h_end = min(h + img_size, H)
            for w in range(0, W, img_size - overlap):
                w_end = min(w + img_size, W)
                gt = raster_arr[h:h_end, w:w_end]

                if gt.shape[:2] != (img_size, img_size):
                    padh = img_size - gt.shape[0]
                    padw = img_size - gt.shape[1]
                    gt = np.pad(gt, ((0, padh), (0, padw)), mode='constant', constant_values=0)

                cv2.imwrite(os.path.join(out_dir_gt, f"{id}_{h}_{w}.png"), gt.astype(np.uint8))


def extract_color_mask(map_dir, out_dir, year, color):
    out_dir_gt = os.path.join(out_dir, 'corrected_gt')
    os.makedirs(out_dir_gt, exist_ok=True)
    for id in MAP_IDS:
        print(id)
        mapfiles = glob(os.path.join(map_dir, f'{id}_*.tif'))
        years = np.array([int(os.path.basename(x).split('.')[0].split('_')[1]) for x in mapfiles])
        idx = np.argmin(np.abs(years - year))
        nearest_year = years[idx]
        map_array = read_tiff(mapfiles[idx])
        mask = (map_array == np.array(color).reshape(1, 1, 3)).all(axis=-1).astype(np.float32)

        kernel_size = 25
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = np.stack([mask] * 3, axis=-1) * 255

        filename = f"{id}_{nearest_year}.png"
        cv2.imwrite(os.path.join(out_dir_gt, filename), mask)


def tiff_to_png(map_dir):
    for id in MAP_IDS:
        print(id)
        mapfiles = glob(os.path.join(map_dir, f'{id}_*.tif'))
        for mf in mapfiles:
            map_array = read_tiff(mf).astype(np.uint8)
            map_array = np.stack([map_array] * 3, axis=-1)
            cv2.imwrite(mf.replace('tif', 'png'), map_array)


def full_gt_map(map_dir, raster_dir, out_dir, category, colors, year, use_morph=False):
    for id in MAP_IDS:
        print(id)
        mapfiles = glob(os.path.join(map_dir, f'{id}_*.tif'))
        years = np.array([int(os.path.basename(x).split('.')[0].split('_')[1]) for x in mapfiles])
        idx = np.argmin(np.abs(years - year))
        nearest_year = years[idx]
        map_array = read_tiff(mapfiles[idx])
        rasterfile = os.path.join(raster_dir, f'{id}_2023_{category}.tif')
        if os.path.exists(rasterfile):
            raster_array = (read_tiff(rasterfile) < 255).astype(float)
            H = min(map_array.shape[0], raster_array.shape[0])
            W = min(map_array.shape[1], raster_array.shape[1])
        else:
            H, W = map_array.shape[:2]

        mask = np.zeros_like(map_array[..., 0])
        for c in colors:
            mask = np.logical_or(mask, np.abs(map_array - np.array(c).reshape(1, 1, 3)).sum(axis=-1) < 15)
        mask = mask.astype(float)

        if use_morph:
            kernel_size = 25
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        gt_map = np.zeros_like(map_array)
        gt_map[..., 0] = mask * 255
        if os.path.exists(rasterfile):
            gt_map[:H, :W, 1] = raster_array[:H, :W] * 255

        filename = f"{id}_{nearest_year}_{category}.png"
        cv2.imwrite(os.path.join(out_dir, filename), gt_map)


def merge_road(map_dir, out_dir, category):
    mapfiles = glob(os.path.join(map_dir, f'*_{category}.png'))
    for mf in mapfiles:
        print(mf)
        map_array = cv2.imread(mf)
        map_array_0 = map_array[:, :, 0]
        map_array_1 = map_array[:, :, 1]
        road_mask = cv2.imread(mf.replace(category, 'road'))[..., 0]
        mask = (map_array_0 == 0) & (map_array_1 == 255) & (road_mask == 255)
        map_array_0[mask] = 255

        kernel_size = 25
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        map_array_0 = cv2.morphologyEx(map_array_0, cv2.MORPH_CLOSE, kernel)

        map_array[..., 0] = map_array_0
        cv2.imwrite(os.path.join(out_dir, mf.replace(category, f'{category}_r')), map_array)


def generate_4cls_gt(in_dir, out_dir):
    files = glob(os.path.join(in_dir, '*for_r.png'))
    patches = [os.path.basename(f).replace('_for_r.png', '') for f in files]
    categories = ['agr', 'for', 'wat', 'set']
    for p in patches:
        gt = []
        for i, cat in enumerate(categories):
            img = cv2.imread(os.path.join(in_dir, f'{p}_{cat}_r.png'), cv2.IMREAD_UNCHANGED)[..., 0]
            mask = (img == 255).astype(float) * (i + 1)
            gt.append(mask)
        gt = np.stack(gt, axis=-1).max(axis=-1)
        cv2.imwrite(os.path.join(out_dir, f'{p}.png'), gt)


def label_map_to_rgb(map_dir, out_dir):
    rgb = [
        [0, 0, 0],
        [254, 252, 233],
        [198, 211, 137],
        [191, 228, 249],
        [247, 187, 164]
    ]
    mapfiles = glob(os.path.join(map_dir, '*.png'))
    for mf in mapfiles:
        img = cv2.imread(mf, cv2.IMREAD_UNCHANGED)
        rbg_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for i, color in enumerate(rgb):
            rbg_img[img==i] = np.array(color)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(mf)), rbg_img[:, :, ::-1])


if __name__=="__main__":
    # path = "/home/yuan/data/HisMap/Hameln"
    path = "/media/yuan/luna/data/HisMAP/Hameln"

    # label_map_to_rgb(
    #     "/home/yuan/data/HisMap/Hameln/raster_rec",
    #     "/home/yuan/data/HisMap/Hameln/raster_rgb"
    # )

    # generate_4cls_gt(f"{path}/raster_gt", f"{path}/raster_rec")

    # generate_4cls_dataset(
    #     f"{path}/TK_projektiv_entzerrt_1m",
    #     f"{path}/Raster_2023",
    #     f"{path}/seg4cls_train",
    #     ['agr', 'for', 'wat', 'set'],
    #     None,
    # )

    generate_4cls_pseudo_labels(
        f"{path}/exp/seg4cls_2016/epoch20/holistic_seg_result",
        f"{path}/seg4cls_train/gt_pseudo_2011",
        ['agr', 'for', 'wat', 'set'],
    )

    # extract_color_mask(
    #     "/home/data/HisMAP/Hameln/TK_projektiv_entzerrt_1m",
    #     "/home/data/HisMAP/Hameln/seg_forest",
    #     2017,
    #     [198, 211, 137]
    # )

    # tiff_to_png("/home/data/HisMAP/Hameln/Raster_2023")
    color_dict = {
        'for': [[198, 211, 137]],
        'wat': [[191, 228, 249]],  # [0, 157, 226],
        'set': [[247, 187, 164], [226, 0, 14], [211, 211, 210], [116, 115, 106]],
        'agr': [[254, 252, 233], [231, 239, 211]],
        'road': [[255, 255, 255], [249, 177, 0], [254, 236, 0]]
    }


    # for k, v in color_dict.items():
    #     full_gt_map(
    #         f"{path}/TK_projektiv_entzerrt_1m",
    #         f"{path}/Raster_2023",
    #         f"{path}/raster_gt",
    #         k,
    #         v,
    #         2017,
    #         False
    #     )


    # for k in color_dict.keys():
    #     merge_road(
    #         f"{path}/raster_gt",
    #         f"{path}/raster_rec",
    #         k
    #     )


