import numpy as np
import cv2
import time

def overlay_mask(image, mask, alpha=0.3, color=(0,0,128)):
    
    assert mask.ndim==2 and image.shape[:2] == mask.shape , "Check the shape of image and mask"
    start_time = time.time()
    # Ensure consistent data types
    overlayed = image.astype(np.uint8)
    mask_colored = np.zeros_like(image, dtype=np.uint8)
    print(time.time()-start_time, "prepare")
    start_time = time.time()
    for i in range(3):
        mask_colored[:, :, i] = color[i]
    print(time.time()-start_time, "color")
    mask = mask.astype(bool)
    start_time = time.time()
    # Apply addWeighted to the entire images, then mask
    overlayed = cv2.addWeighted(overlayed, 1 - alpha, mask_colored, alpha, 0)
    print(time.time() - start_time , "blend")
    start_time = time.time()
    # overlayed[mask] = blended[mask]
    print(time.time() - start_time , "write")
    
    return overlayed