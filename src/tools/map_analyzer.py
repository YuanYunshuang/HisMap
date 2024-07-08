import os
import cv2
import numpy as np
import json
import open3d as o3d
import torch
import torch.nn.functional as F


def plot_color_space(colors):
    pcd = o3d.geometry.PointCloud()

    # Assign the points and colors
    pcd.points = o3d.utility.Vector3dVector(colors)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


def analyze_color_coding(map_file):
    img = cv2.imread(map_file)
    h, w = img.shape[:2]
    colors = np.unique(img[::4, ::4].reshape(-1, 3), axis=0)
    # plot_color_space(colors.astype(float) / 255)
    np.sort(colors, axis=0)
    sample_size = 128
    color_encoding_file = "../../data/color_encoding.json"
    if os.path.exists(color_encoding_file):
        with open(color_encoding_file, 'r') as f:
            color_encoding = json.load(f)
    else:
        color_encoding = {}

    try:
        for i, c in enumerate(colors):
            mask = (img == c.reshape(1, 1, 3)).all(axis=-1)
            n_pix = np.count_nonzero(mask)
            samples = np.where(mask)
            while True:
                select_idx = np.random.randint(0, n_pix)
                x, y = samples[0][select_idx], samples[1][select_idx]
                if sample_size < x < h - sample_size and sample_size < y < w - sample_size:
                    # mark sample with a red rectangle:
                    sample = img[x-sample_size-1:x+sample_size, y-sample_size-1:y+sample_size]
                    # cross = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]]) + sample_size
                    # sample[cross[:, 0], cross[:, 1]] = 0
                    cv2.rectangle(sample, (sample_size - 3, sample_size - 3),
                                 (sample_size + 3, sample_size + 3), (0, 0, 255), 2)
                    color_bar = np.zeros((20, sample_size * 2 + 1, 3), dtype=np.uint8) + c.reshape(1, 1, 3)
                    sample = np.concatenate([sample, color_bar], axis=0)
                    cv2.imwrite("/home/data/HisMAP/full_map/tmp.png", sample)

                    print(f"Type in the info of the color{i:d} ({c[0]}, {c[1]}, {c[2]}):")
                    # rgba = input("RBGA(int, int, int, float):")
                    rgb = ','.join([str(ci) for ci in c])
                    layer = input("Layer:")
                    color_encoding[rgb] = layer
                    break
    except:
        with open(color_encoding_file, 'w') as f:
            json.dump(color_encoding, f)
    finally:
        with open(color_encoding_file, 'w') as f:
            json.dump(color_encoding, f)


def draw_color_palette(map_file):
    img = cv2.imread(map_file)
    h, w = img.shape[:2]
    colors = np.unique(img[::4, ::4].reshape(-1, 3), axis=0)
    # colors = colors[(colors>50).any(axis=-1)]
    np.sort(colors, axis=0)

    # Parameters
    num_cols = 10
    num_colors = len(colors)
    num_rows = (num_colors + num_cols - 1) // num_cols  # Ensure we cover all colors

    palette_width = 100
    palette_height = 50
    margin = 10
    text_offset = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1

    # Create an image canvas
    image_height = (palette_height + text_offset + margin) * num_rows + margin
    image_width = (palette_width + margin) * num_cols + margin
    image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255  # White background

    # Draw each color and its RGB value
    for i, color in enumerate(colors):
        row = i // num_cols
        col = i % num_cols
        top_left = (margin + col * (palette_width + margin), margin + row * (palette_height + text_offset + margin))
        bottom_right = (top_left[0] + palette_width, top_left[1] + palette_height)
        cv2.rectangle(image, top_left, bottom_right, color.tolist(), -1)

        # Prepare the RGB value text
        rgb_text = f"{color}"
        text_position = (top_left[0], bottom_right[1] + text_offset - 5)
        cv2.putText(image, rgb_text, text_position, font, font_scale, (0, 0, 0), font_thickness)

    # Show the image
    cv2.imshow("Color Palette", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_color(map_file, color, out_file):
    img = cv2.imread(map_file)
    color = np.array(color).reshape(1, 1, 3)
    mask = (img == color).all(axis=-1).astype(np.float32)

    kernel_size = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(out_file, mask.astype(np.uint8) * 255)


def extract_boundary(map_file, out_file):
    img = cv2.imread(map_file) / 255
    img = torch.from_numpy(img[..., 0:1]).permute(2, 0, 1).unsqueeze(0).float()
    filters = torch.ones(1, 1, 3, 3)
    boundary = F.conv2d(img, filters, padding=1)
    boundary = torch.clip(1 - (boundary - 9 / 2).abs() / 4.5, min=0, max=0.5) * 2
    boundary = boundary.float().repeat(1, 3, 1, 1).cpu().numpy()
    boundary = boundary.squeeze(0).transpose(1, 2, 0) * 255
    cv2.imwrite(out_file, boundary.astype(np.uint8))


if __name__=="__main__":
    map_file = "/home/data/HisMAP/full_map/3922_2017.tif"
    # extract_color(map_file, [137, 211, 198], "/home/data/HisMAP/full_map/3922_2017_forest.tif")
    # extract_boundary("/home/data/HisMAP/full_map/3922_2017_forest.tif",
    #                  "/home/data/HisMAP/full_map/3922_2017_forest_bdry.tif")

    img = cv2.imread("/home/yuan/Downloads/3922_2017_forest_bdry.tif")
    cv2.imwrite("/home/yuan/Downloads/tmp.png", img[256:512, 256:512])

