import math
from typing import List
import numpy as np



def make_image_grid(images: List[np.ndarray]) -> np.ndarray:
    # Tile images (RGB) into a nearly square grid.
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    base_h, base_w = images[0].shape[:2]
    cols = int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / cols))
    grid = np.zeros((rows * base_h, cols * base_w, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        if img.shape[:2] != (base_h, base_w):
            import cv2

            img = cv2.resize(img, (base_w, base_h))
        r, c = divmod(idx, cols)
        grid[r * base_h : (r + 1) * base_h, c * base_w : (c + 1) * base_w] = img
    return grid