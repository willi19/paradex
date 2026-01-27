import numpy as np
import cv2
import torch

def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.5) -> np.ndarray:
    # Alpha-blend binary mask onto RGB image.
    mask = (mask > 0.5).astype(np.float32)
    overlay = image.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)[None, None, :]
    overlay = overlay * (1 - alpha * mask[..., None]) + color_arr * (alpha * mask[..., None])
    return overlay.astype(np.uint8)
