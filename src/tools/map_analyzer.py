import os
import cv2
import numpy as np
import json
import open3d as o3d


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


if __name__=="__main__":
    analyze_color_coding("/home/data/HisMAP/full_map/3922_2017.tif")

