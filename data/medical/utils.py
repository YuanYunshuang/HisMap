
from glob import glob
import numpy as np
from PIL import Image



def med_img_to_rgb_pil_image(img):
    """
    Convert medical grayscale image to RGB image. Syntra model takes 3-channel images as input, 
    with each channel being 8 bits. ADAC images are single-channel images with the maximum 
    pixel value larger than 255. To keep the accuracy of the original ADAC images, we encode 
    the pixel values into the three RGB 8-bit channels.

    Parameters
    ----------
    img: np.array
        Grayscale image array. Shape (H, W) 

    Returns
    -------
    pil_img: PIL.Image
        RGB image.
    """
    # encode pixel value to 3 8-bit channels
    v = img.astype(np.uint32)
    if (v.max() >= (1 << 24)) or (v.min() < 0):
        raise ValueError("Values must be in [0, 2^24-1] for 3x8-bit RGB lossless packing.")
    r = (v & 0xFF).astype(np.uint8)
    g = ((v >> 8) & 0xFF).astype(np.uint8)
    b = ((v >> 16) & 0xFF).astype(np.uint8)
    rgb_array = np.stack([r, g, b], axis=-1)
    pil_img = Image.fromarray(rgb_array, mode='RGB')
    return pil_img


def rgb_img_to_med_img(pil_img):
    """
    Convert RGB image to medical grayscale image.
    Parameters
    ----------
    pil_img: PIL.Image
        RGB image.

    Returns
    -------
    img: np.array
        Grayscale image array. Shape (H, W) with min 0 and max 2**24-1
    """
    rgb_array = np.array(pil_img)
    r = rgb_array[..., 0].astype(np.uint32)
    g = rgb_array[..., 1].astype(np.uint32)
    b = rgb_array[..., 2].astype(np.uint32)
    img = r + (g << 8) + (b << 16)
    return img