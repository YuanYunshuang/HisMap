import cv2
import numpy as np


def paint_stroke(f, width=128):
    """
    Draw the stroke f on a blank image.

    :param f: stroke parameterized with Bezier Curve on points
        (x0, y0), (x1, y1), (x2, y2), the radius of paint circle
        (z0, z1) and the color of the stroke (w0, w1).
    :param width: canvas width in pixel
    :return: drawn image of shape (width, width)
    """
    def normal(x, width):
        return (int)(x * (width - 1) + 0.5)
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
    canvas = np.zeros([width * 2, width * 2]).astype('float32')
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
