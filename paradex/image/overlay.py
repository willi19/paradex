import numpy as np
import cv2
import torch

def overlay_mask(image, mask, color, alpha=0.7):
    assert mask.ndim==2 and image.shape[:2] == mask.shape , "Check the shape of image and mask"
    color = np.array(color, dtype=image.dtype)  # 추가!
    image[mask] = image[mask] * alpha + color * (1-alpha)
    return image
