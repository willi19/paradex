import numpy as np
import cv2
import torch

def overlay_mask(image, mask, color, alpha=0.7):
    """Alpha-blend a solid color over the masked region, in place.

    Parameters
    ----------
    image : numpy.ndarray
        ``(H, W, 3)`` image; **modified in place** (copy first to keep the original).
    mask : numpy.ndarray
        Boolean ``(H, W)`` mask matching the image size.
    color : sequence of int
        BGR color painted where ``mask`` is True.
    alpha : float
        Weight of the original image at masked pixels (``alpha`` image +
        ``1 - alpha`` color).

    Returns
    -------
    numpy.ndarray
        The same ``image`` array, for convenience.
    """
    assert mask.ndim==2 and image.shape[:2] == mask.shape , "Check the shape of image and mask"
    color = np.array(color, dtype=image.dtype)  # 추가!
    image[mask] = image[mask] * alpha + color * (1-alpha)
    return image
