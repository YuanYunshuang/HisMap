import numpy as np
import cv2
import matplotlib.pyplot as plt


def overlay_label_on_image(image, label, color=(0, 255, 0), alpha=0.5):
    """
    Overlay a mask on an RGB image.

    Parameters:
    - image: np.ndarray - RGB image (H, W, 3)
    - label: np.ndarray - boolean or uint8 mask (H, W)
    - color: tuple - RGB color for the mask overlay (default is green)
    - alpha: float - Transparency factor for the overlay (0 to 1)

    Returns:
    - overlayed_image: np.ndarray - Image with mask overlay
    """
    # Ensure image and mask sizes match
    if image.shape[:2] != label.shape[:2]:
        raise ValueError("Image and mask must have the same height and width")

    # Make sure the image is in RGB (OpenCV loads in BGR by default)
    if image.shape[2] == 3 and np.max(image) > 1.0:
        image = image.astype(np.uint8)

    # Create a color mask
    color_mask = np.zeros_like(image)
    if label.dtype == bool:
        color_mask[label > 0] = color
    elif np.max(label) > 1:
        assert isinstance(color, list) and isinstance(color[0], list)
        for l in range(1, np.max(label) + 1):
            color_mask[label == l] = color[l-1]

    # Blend the images
    overlayed_image = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

    return overlayed_image


def apply_colormap_transparent(image, mask, cmap='jet'):
    """
    Apply a transparent Jet colormap to an image using a float mask.

    Parameters:
    - image (np.ndarray): Input image (H, W, 3), RGB, dtype=uint8
    - mask (np.ndarray): Float mask (H, W), values in [0.0, 1.0]

    Returns:
    - np.ndarray: Image with Jet colormap blended over original image
    """
    # Ensure mask is float32 and clipped between 0 and 1
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)

    # Resize if needed
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask must have the same height and width")

    # Convert mask to Jet colormap using matplotlib
    jet_map = plt.get_cmap(cmap)
    jet_colors = jet_map(1 - mask)[..., :3] # Ignore alpha, get RGB only
    # jet_colors[..., -1] = 0.6
    jet_colors = (jet_colors * 255).astype(np.uint8)

    # Blend the jet colormap with the original image using the mask as alpha
    blended = image.copy().astype(np.float32)
    jet_colors = jet_colors.astype(np.float32)

    for c in range(3):
        blended[:, :, c] = (1 - 0.5) * blended[:, :, c] + 0.5 * jet_colors[:, :, c]

    return blended.astype(np.uint8)

