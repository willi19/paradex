import numpy as np
import cv2

def overlay_mask(image, mask, alpha=0.3, color=(0,0,128)):
    
    assert mask.ndim==2 and image.shape[:2] == mask.shape , "Check the shape of image and mask"
    overlayed  = np.array(image)
    mask_colored = np.zeros_like(image, dtype=np.uint8)
    for i in range(3):
        mask_colored[:, :, i] = color[i]
    mask = mask.astype(bool)
    overlayed[mask] = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)[mask]
    
    return overlayed