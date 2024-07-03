import numpy as np
import matplotlib.pyplot as plt
import cv2


def draw_bezier_curve(bezier_curve, nt=128, show=False):
    """
    Draw quadratic-bezier curve with the four-point parameters.

    :param bezier_curve: (4, 2)
    :param nt: (int) number of drawing points.
    """
    ts = np.linspace(0, 1, nt)
    ts = np.stack((np.ones_like(ts), ts, ts**2, ts**3), axis=1)
    characteristic_mat = np.array([
        [1, 0, 0, 0],
        [-3, 3, 0, 0],
        [3, -6, 3, 0],
        [-1, 3, -3, 1]
    ])
    curve = ts @ (characteristic_mat @ bezier_curve)
    if show:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(curve[:, 0], curve[:, 1], '-k')
        ax.plot(bezier_curve[:, 0], bezier_curve[:, 1], 'og--')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.show()
        plt.close()

    return curve


def draw_splines(mask, params, meter_per_pix=1, nt=32, show=False):
    """
    Draw splines.

    :param mask: (H, W), pixels with True are the valid control points of splines.
    :param params: (H, W, 8, 3), parameters (direction, length) of each possible control point
        relative to the 8 neighbors.
    :param meter_per_pix: (int) resize factor.
    """
    h, w = mask.shape
    nbrs_next = np.array([[-1, 1], [0, 1], [1, 1], [1, 0]])
    inds = np.stack(np.where(mask), axis=1)
    points = []
    curves = []
    thickness = []
    for idx in inds:
        nbrs = nbrs_next + idx
        valid = (nbrs >= 0).all(-1) & (nbrs[:, 0] < h) & (nbrs[:, 1] < w)
        valid_idx = np.where(valid)[0]
        valid_nbrs = nbrs[valid_idx]
        p1 = (idx + 0.5) * meter_per_pix
        for i, p4_idx in zip(valid_idx, valid_nbrs):
            p4 = (p4_idx + 0.5) * meter_per_pix
            p4_param = params[p4_idx[0], p4_idx[1], i+4]
            p1_param = params[idx[0], idx[1], i]
            p2 = p1_param[0] * p4_param[1:3] * meter_per_pix + p1
            p3 = p4_param[0] * p4_param[1:3] * meter_per_pix + p4
            pts = np.stack([p1, p2, p3, p4])
            points.append(pts)
            curves.append(draw_bezier_curve(pts, nt=nt))
            thickness.append([p1_param[-1] * meter_per_pix / 2, p4_param[-1] * meter_per_pix / 2])

    if show:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim([0, h * meter_per_pix])
        ax.set_ylim([0, w * meter_per_pix ])
        for pts, curve, thick in zip(points, curves, thickness):
            ax.plot(curve[:, 0], curve[:, 1], '-k')
            ax.plot(pts[:, 0], pts[:, 1], '--')
        plt.show()
        plt.close()
    return points, curves, thickness


def draw_splines_on_canvas(mask, params, resize_factor=1, nt=32, show=False):
    points, curves, thickness = draw_splines(mask, params, resize_factor, nt=nt)
    # curves = curves.astype(int)
    h, w = mask.shape
    canvas = np.zeros((h * resize_factor, w * resize_factor), dtype=np.float32)
    for curve, thick in zip(curves, thickness):
        curve = curve.astype(np.int32)
        line_width = np.linspace(thick[0], thick[1], len(curve)).astype(int)
        line_width = np.clip(line_width, 1, resize_factor // 2)
        for i in range(0, len(curve) - 1):
            cv2.line(canvas, tuple(curve[i]), tuple(curve[i+1]), color=1, thickness=line_width[i])

    if show:
        cv2.imshow('canvas', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return canvas


if __name__ == '__main__':
    # draw_bezier_curve(np.random.random((4, 2)), nt=128, show=True)
    grid_mask = np.random.rand(16, 16) > 0.7
    grid_mask[1, 1] = True
    grid_params = np.random.rand(16, 16, 8, 4)
    draw_splines_on_canvas(grid_mask, grid_params, resize_factor=16, show=True)