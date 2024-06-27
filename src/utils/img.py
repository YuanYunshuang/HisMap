import torch


def img_coord(width):
    x = torch.arange(width)
    coord = torch.stack(torch.meshgrid([x, x], indexing='ij'), dim=0) / 127.

    return coord


if __name__=="__main__":
    img_coord(128)