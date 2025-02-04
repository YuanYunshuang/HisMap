import glob
import os
import argparse
import shutil
import sys

import cv2
import numpy as np
from PIL import Image

from src.tools.map_analyzer import read_tiff
from src.tools.lgln import LGLN_COLOR

def parse_and_copy_valid_maps(path_in, path_out):
    label_files = glob.glob(os.path.join(path_in, '*/label/*.png'))
    mapfiles = []
    labelfiles = []
    for f in label_files:
        path, basename = os.path.split(f)
        split = path.split('/')[-2]
        patch, year = basename.split('.')[0].split('_')
        # label = cv2.imread(f, cv2.IMREAD_GRAYSCALE) / 50
        filename = os.path.join(path_in, split, 'maps', f"{patch}_{year}.tif")
        if not os.path.exists(filename):
            filename = os.path.join(path_in, split, 'label', f"{patch}_{int(year) - 1:d}.tif")
        if not os.path.exists(filename):
            filename = os.path.join(path_in, split, 'label', f"{patch}_{int(year) + 1:d}.tif")
        if not os.path.exists(filename):
            continue
        shutil.copy(filename, os.path.join(path_out, 'maps', f"{patch}_{year}.tif"))
        shutil.copy(f, os.path.join(path_out, 'labels', f"{patch}_{year}.png"))
        mapfiles.append(filename)
        labelfiles.append(f)
    return mapfiles, labelfiles


def crop(sheet_file, label_file, out_dir, crop_size, scale=0.5, use_label=False, num_pixel_present=100):
    print(f"Cropping...{sheet_file}")
    out_dir_img = os.path.join(out_dir, 'images', f"{crop_size:d}_{scale:.2f}")
    out_dir_lbl = os.path.join(out_dir, 'labels', f"{crop_size:d}_{scale:.2f}")
    os.makedirs(out_dir_img, exist_ok=True)
    os.makedirs(out_dir_lbl, exist_ok=True)
    TIF = read_tiff(sheet_file)
    TIF = cv2.resize(TIF, None, fx=scale, fy=scale)
    LABEL = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE) / 50
    L_RGB = np.zeros((LABEL.shape[0], LABEL.shape[1], 3), dtype=np.uint8)
    for l in range(1, 6):
        L_RGB[LABEL == l] = LGLN_COLOR[l-1]
    L_RGB = cv2.resize(L_RGB, None, fx=scale, fy=scale)
    crop_size = int(crop_size * scale)
    basename = os.path.basename(sheet_file)
    patch, year = basename.split('.')[0].split('_')

    h, w = TIF.shape[0], TIF.shape[1]
    x = np.arange(0, h, crop_size)
    y = np.arange(0, w, crop_size)
    xs, ys = np.meshgrid(x, y)
    xs, ys = xs.flatten(), ys.flatten()
    for x1, y1 in zip(xs, ys):
        x2, y2 = x1 + crop_size, y1 + crop_size
        x2 = h if x2 > h else x2
        y2 = w if y2 > w else y2
        M = TIF[y1:y2, x1:x2]
        L = L_RGB[y1:y2, x1:x2]
        l_cls = '_'
        if use_label:
            for i in range(1, 5):
                if i == 4 and np.sum(L >= i) > num_pixel_present:
                    l_cls += '1'
                elif np.sum(L == i) > num_pixel_present:
                    l_cls += '1'
                else:
                    l_cls += '0'
        cv2.imwrite(os.path.join(out_dir_img, f"{patch}_{year}_{x1}_{y1}{l_cls}.jpg"), M)
        cv2.imwrite(os.path.join(out_dir_lbl, f"{patch}_{year}_{x1}_{y1}.jpg"), L)


def crop_series(sheet_files, label_files, crop_size, scale=0.5, use_label=False):
    for sf, lf in zip(sheet_files, label_files):
        crop(sf, lf,
             "/home/yuan/data/HisMap/LGLN/dataset/llm_label",
             crop_size, scale, use_label)


def draw_grid(image, rows, cols, color=(255, 0, 0), thickness=2):
    h, w = image.shape[0], image.shape[1]
    row_spacing = h // rows
    col_spacing = w // cols
    s = thickness // 2

    for r in range(1, rows):
        y = r * row_spacing
        cv2.line(image, (0, y), (w, y), color, thickness)
    for c in range(1, cols):
        x = c * col_spacing
        cv2.line(image, (x, 0), (x, h), color, thickness)

    return image

def draw_title(image, title_text, font_scale=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    bg_color = (255, 255, 255)  # White text
    text_color = (0, 0, 0)  # Black background for title

    # Get image dimensions
    height, width, channels = image.shape

    # Define title bar height
    title_bar_height = 50  # Adjust based on text size

    # Create a blank strip (same width as image, but shorter in height)
    title_bar = np.full((title_bar_height, width, 3), bg_color, dtype=np.uint8)

    # Get text size for centering
    (text_width, text_height), _ = cv2.getTextSize(title_text, font, font_scale, font_thickness)
    text_x = (width - text_width) // 2  # Centered horizontally
    text_y = (title_bar_height + text_height) // 2  # Centered vertically

    # Write the text on the title bar
    cv2.putText(title_bar, title_text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    # Concatenate title bar and image vertically
    output_image = np.vstack((title_bar, image))
    return output_image

def resize_keep_aspect(img, height):
    aspect_ratio = img.shape[1] / img.shape[0]
    new_width = int(height * aspect_ratio)
    return cv2.resize(img, (new_width, height))

def stack_images(images, spacing=10):
    height = max([x.shape[0] for x in images])
    width = max([x.shape[1] for x in images])
    for i in range(len(images)):
        img = images[i]
        pad_h = height - img.shape[0]
        pad_w = width - img.shape[1]
        images[i] = np.pad(images[i], ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    # Create a blank separator (same height as images, width = spacing)
    height, width, channels = images[0].shape
    separator = np.ones((height, spacing, channels), dtype=np.uint8) * 255  # White space

    # Concatenate images with spacing
    patches = []
    for x in images:
        patches.append(x)
        patches.append(separator)
    output_image = np.hstack(patches)
    return output_image


def image_side_by_side(image1, image2):
    # Get the dimensions of both images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Determine the new height (max of both images) and total width
    new_height = max(height1, height2)
    total_width = width1 + width2

    # Create a new blank image with the appropriate size
    new_image = Image.new('RGB', (total_width, new_height))

    # Paste the two images into the new image
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (width1, 0))
    return new_image


def prepare_prompt_img(sheet_files, out_dir, label, crop_size, scale=1/3):
    os.makedirs(out_dir, exist_ok=True)
    years = []
    TIFs = []
    for sheet_file in sheet_files:
        year = os.path.basename(sheet_file).split('.')[0].split('_')[-1]
        if year != '2023':
            TIF = read_tiff(sheet_file)
        else:
            TIF = np.array(Image.open(sheet_file).convert('RGB'))
        TIFs.append(TIF)
        years.append(year)
    TIFs.append(label)
    years.append(year)

    h, w = TIFs[0].shape[0], TIFs[0].shape[1]
    x = np.arange(0, h, crop_size)
    y = np.arange(0, w, crop_size)
    xs, ys = np.meshgrid(x, y)
    xs, ys = xs.flatten(), ys.flatten()
    for x1, y1 in zip(xs, ys):
        x2, y2 = x1 + crop_size, y1 + crop_size
        x2 = h if x2 > h else x2
        y2 = w if y2 > w else y2
        imgs = []
        for year, TIF in zip(years, TIFs):
            M = TIF[y1:y2, x1:x2]
            if M.ndim == 2:
                M = M[..., np.newaxis].repeat(3, axis=2)
            M = cv2.resize(M, None, fx=scale, fy=scale)
            M = draw_grid(M, 2, 2, color=(0, 255, 0), thickness=4)
            M = cv2.cvtColor(M, cv2.COLOR_BGR2RGB)
            M = draw_title(M, year)
            imgs.append(M)

        img = stack_images(imgs)
        cv2.imwrite(os.path.join(out_dir, f"{x1}_{y1}.jpg"), img)


def prepare_promt_img_with_legend(lendfile, img_dir, out_dir):
    img_files = glob.glob(f"{img_dir}/*.jpg")
    image1 = Image.open(lendfile)
    for img_file in img_files:
        basename = os.path.basename(img_file)
        year = basename.split('.')[0].split('_')[1]
        if not "197" in year:
            continue
        image2 = Image.open(img_file)
        new_image = image_side_by_side(image1, image2)
        new_image.save(os.path.join(out_dir, os.path.basename(img_file)))



if __name__ == '__main__':
    # sheet_files, label_files = parse_and_copy_valid_maps(
    #     "/home/yuan/data/HisMap/LGLN/dataset/age_tracing_data",
    #     "/home/yuan/data/HisMap/LGLN/dataset/llm_label"
    # )

    prepare_promt_img_with_legend(
        "/home/yuan/Downloads/hismap_legend_2.png",
        "/home/yuan/data/HisMap/LGLN/dataset/attn/train",
        "/home/yuan/data/HisMap/LGLN/dataset/llm_label/prompt_img/197x"
    )

    path_in = "/home/yuan/data/HisMap/LGLN/dataset/llm_label"
    sheet = 3821
    sheet_files = sorted(glob.glob(os.path.join(path_in, f'maps/{sheet}_*.tif')))
    label_file = os.path.join(path_in, f'labels/{sheet}_2023.png')
    LABEL = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE) / 50
    LABEL_RGB = np.zeros((LABEL.shape[0], LABEL.shape[1], 3), dtype=np.uint8)
    for l in range(1, 6):
        LABEL_RGB[LABEL == l] = LGLN_COLOR[l-1]
    prepare_prompt_img(
        sheet_files,
        os.path.join(path_in, 'prompt_img', str(sheet)),
        LABEL_RGB,
        1152
    )
