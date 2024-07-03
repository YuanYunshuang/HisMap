import random
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


def normal(x, width):
    return (int)(x * (width - 1) + 0.5)


def paint_stroke(f, width=128):
    """
    Draw the stroke f on a blank image.

    :param f: stroke parameterized with Bezier Curve on points
        (x0, y0), (x1, y1), (x2, y2), the radius of paint circle
        (z0, z1) and the color of the stroke (w0, w1).
    :param width: canvas width in pixel
    :return: drawn image of shape (width, width)
    """
    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    x0 = normal(x0, width * 2)
    x1 = normal(x1, width * 2)
    x2 = normal(x2, width * 2)
    y0 = normal(y0, width * 2)
    y1 = normal(y1, width * 2)
    y2 = normal(y2, width * 2)
    z0 = (int)(1 + z0 * width // 2)
    z2 = (int)(1 + z2 * width // 2)
    canvas = np.zeros([width * 2, width * 2]).astype(float)
    tmp = 1. / 100
    for i in range(100):
        t = i * tmp
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        z = (int)((1-t) * z0 + t * z2)
        w = (1-t) * w0 + t * w2
        cv2.circle(canvas, (y, x), z, w, -1)
    return 1 - cv2.resize(canvas, dsize=(width, width))


def paint_strokes(strokes, canvas, renderer):
    """
    Paint strokes on the given canvas.

    :param strokes: stroke parameters with shape (B, N*13,) or (B, N, 13),
        where N is the number of strokes.
    :param canvas: in shape (3, H, W).
    :param renderer: renderer that converts stroke parameters to images.
    :return: painted canvas.
    """
    x = strokes.view(-1, 10 + 3)
    stroke = 1 - renderer(x[:, :10])
    stroke = stroke.view(-1, 128, 128, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 5, 1, 128, 128)
    color_stroke = color_stroke.view(-1, 5, 3, 128, 128)
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
    return canvas


def paint_polygon(polygon, width=128):
    points = polygon.reshape(-1, 2)
    centroid = np.mean(points, axis=0)

    # Sort the points based on their angle relative to the centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_points = (points[np.argsort(angles)] * width * 2).astype(int)
    # sorted_points = np.clip(sorted_points, a_min=0, a_max=width * 2)

    # draw the polygon on image
    canvas = np.zeros([width * 2, width * 2], dtype=float)
    cv2.fillPoly(canvas, [sorted_points], color=1)
    return cv2.resize(canvas, dsize=(width, width))


def paint_polygons_by_param(polygons, width=128, show=False):
    canvas = np.zeros([width, width, 3], dtype=float)
    for polygon in polygons:
        mask = paint_polygon(polygon[:-3], width).astype(bool)
        canvas[mask] = polygon[-3:]
    if show:
        cv2.imshow('img', canvas)
        cv2.waitKey()
    return canvas


def paint_polygons_by_renderer(polygons, width=128, renderer=None):
    """
    Paint polygons on the canvas.

    :param polygons: batch of polygons in shape (B, N, 2n+3).
    :param width: canvas width in pixel.
    :param renderer: renderer that converts polygons to images.
    """
    B, N, _ = polygons.shape
    canvas = torch.zeros([B, 3, width, width], device=polygons.device, dtype=polygons.dtype)

    if renderer.__class__.__name__ == 'ShapeRenderer':
        canvas = renderer(polygons)
    else:
        masks = renderer(polygons[..., :-3].view(B*N, -1))
        masks = masks.view(-1, width, width, 1)
        colored_masks = masks * polygons[..., -3:].view(-1, 1, 1, 3)
        masks = masks.permute(0, 3, 1, 2).view(-1, N, 1, width, width)
        colored_masks = colored_masks.permute(0, 3, 1, 2).view(-1, N, 3, width, width)
        for i in range(N):
            canvas = canvas * (1 - masks[:, i]) + colored_masks[:, i]
    # cv2.imshow('img1', canvas[0].permute(1, 2, 0).detach().cpu().numpy())
    # cv2.waitKey()
    # paint_polygons_by_param(polygons[0].detach().cpu().numpy(), show=True)
    return canvas


def generate_random_polygon(num_points=None, width=128):
    if num_points is None:
        num_points = random.randint(3, 6)
    points = (np.random.rand(num_points, 2) - 0.5) * 2
    centroid = np.mean(points, axis=0)
    color = np.random.rand(3)

    # Sort the points based on their angle relative to the centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_points = (points[np.argsort(angles)] * width).astype(int) + width // 2
    sorted_points = np.clip(sorted_points, a_min=0, a_max=width)
    param = sorted_points.reshape(-1)

    # draw the polygon on image
    mask = np.zeros([width, width], dtype=float)
    cv2.fillPoly(mask, [sorted_points], color=1)
    canvas = mask[..., None] * color.reshape(1, 1, 3)
    # cv2.imshow('img', canvas)
    # cv2.waitKey()
    return param, canvas


def generate_random_circle(width=128):
    param = np.random.randint(0, width, 3)
    color = np.random.randint(0, width, 3)
    canvas = np.zeros([width, width])
    cv2.circle(canvas, param[:2], (param[2] * width), color)
    return param, canvas


def plot_polygon(points):
    plt.figure()
    plt.fill(points[:, 0], points[:, 1], 'b-', alpha=0.3)  # Fill the polygon
    plt.plot(points[:, 0], points[:, 1], 'ro')  # Plot the vertices
    plt.plot(np.append(points[:, 0], points[0, 0]), np.append(points[:, 1], points[0, 1]),
             'b-')  # Connect the vertices
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__=="__main__":
    for i in range(10):
        generate_random_polygon()
