import os
import cv2
import numpy as np
import json
import torch
import torch.nn.functional as F


def plot_color_space(colors):
    import open3d as o3d
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


def angle_vecs(vec1, vec2):
    vec1 = vec1.reshape(-1, 2)
    vec2 = vec2.reshape(-1, 2)
    vec1_norm = np.linalg.norm(vec1, axis=-1, keepdims=True)
    vec2_norm = np.linalg.norm(vec2, axis=-1, keepdims=True)
    return np.arccos((vec1 @ vec2.T) / (vec1_norm * vec2_norm))


def kernel_to_nbrs(kernel_size):
    s = kernel_size // 2
    nbrs = np.meshgrid(np.arange(-s, s + 1), np.arange(-s, s + 1))
    nbrs = np.stack(nbrs, axis=-1).reshape(-1, 2)
    return nbrs


def boundary_to_vector(map_file, kernel_size=5):
    input_img = cv2.imread(map_file)
    img = input_img[..., 0] / 255
    out_img = np.zeros_like(input_img)
    boundry_points = np.stack(np.where(img), axis=-1)
    nbrs = kernel_to_nbrs(kernel_size)
    polygons = []
    i = 1

    while len(boundry_points) > 0:
        point = boundry_points[0]
        img[point[0], point[1]] = 0
        start_point = np.array(point)
        polygon = [start_point]
        while True:
            xs = nbrs[:, 0] + start_point[0]
            ys = nbrs[:, 1] + start_point[1]
            mask = img[xs, ys] > 0
            img[xs, ys] = 0
            if np.sum(mask) == 0:
                if np.linalg.norm(start_point - polygon[0]) < kernel_size:
                    polygon = np.stack(polygon, axis=0)
                    polygons.append(polygon)
                    boundry_points = np.stack(np.where(img), axis=-1)
                    color = (np.random.random(3) * 255).astype(int)
                    cv2.polylines(out_img, [np.stack(polygon, axis=0)[:, ::-1]], True, color.tolist(), 5)
                    cv2.imwrite(map_file.split('.')[0] + '_polygon.jpg', out_img)
                    print(len(boundry_points))
                    break
                else:
                    ks = kernel_size + 2
                    found_next = False
                    while True:
                        tmp_nbrs = kernel_to_nbrs(ks + 2)
                        xs = tmp_nbrs[:, 0] + start_point[0]
                        ys = tmp_nbrs[:, 1] + start_point[1]
                        mask = img[xs, ys] > 0
                        if np.sum(mask) > 0:
                            found_next = True
                            break
                        elif ks < 30:
                            ks += 2
                        else:
                            break
                    if not found_next:
                        polygon = np.stack(polygon, axis=0)
                        polygons.append(polygon)
                        boundry_points = np.stack(np.where(img), axis=-1)
                        color = (np.random.random(3) * 255).astype(int)
                        cv2.polylines(out_img, [np.stack(polygon, axis=0)[:, ::-1]], False, color.tolist(), 5)
                        cv2.imwrite(map_file.split('.')[0] + '_polygon.jpg', out_img)
                        print(len(boundry_points))
                        break

            xs, ys = xs[mask], ys[mask]
            xys = np.stack([xs, ys], axis=-1)

            dxy = xys - start_point
            # dxy_mean = dxy.mean(axis=0, keepdims=True)
            # mean, eigenvectors, eigenvalues = cv2.PCACompute2(xys, mean=xys.mean(axis=0).reshape(-1, 2))
            # main_vec = eigenvectors[np.argmax(eigenvalues)]
            # angles = angle_vecs(dxy, dxy_mean)
            # idx = np.argmin(angles)
            idx = np.argmax(np.linalg.norm(dxy, axis=-1))
            next_point = xys[idx]

            if len(polygon) == 1:
                polygon.append(next_point)
            else:
                last_vec = polygon[-1] - polygon[-2]
                angle_diff = angle_vecs(next_point - polygon[-1], last_vec)
                if angle_diff < 0.3:
                    polygon[-1] = next_point
                else:
                    polygon.append(next_point)
            start_point = next_point
            i += 1
            # if i % 200 == 0:
            #     color = (np.random.random(3) * 255).astype(int)
            #     cv2.polylines(out_img, [np.stack(polygon, axis=0)], True, color.tolist(), 5)
            #     cv2.imwrite(map_file.split('.')[0] + '_polygon.jpg', out_img)


            # vis = img[max(start_point[0] - 128, 0):min(start_point[0] + 128, img.shape[0]),
            #       max(start_point[1] - 128, 0):min(start_point[1] + 128, img.shape[1])]
            # cv2.imshow('img', vis * 255)
            # cv2.waitKey()

    np.savez(map_file.split('.')[0] + '_polygon.npz', polygons=polygons)



if __name__=="__main__":
    map_path = f"/media/yuan/luna/data/HisMAP/full_map"
    # map_file = f"{map_path}/3922_2017.tif"
    # extract_color(map_file, [137, 211, 198], f"{map_path}/3922_2017_forest.tif")
    # extract_boundary(f"{map_path}/3922_2017_forest.tif",
    #                  f"{map_path}/3922_2017_forest_bdry.tif")

    # img = cv2.imread(f"{map_path}/3922_2017_forest_bdry.tif")
    # cv2.imwrite(f"{map_path}/tmp.png", 255 - img[512:1024:, 512:1023])

    boundary_to_vector(f"{map_path}/3922_2017_forest_bdry.tif")

