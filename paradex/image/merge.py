import math
import cv2
import numpy as np

def merge_image(image_dict):
    name_list = sorted(list(image_dict.keys()))
    num_images = len(name_list)
    
    grid_cols = math.ceil(math.sqrt(num_images))
    grid_rows = math.ceil(num_images / grid_cols)
    border_px = 20
    
    new_W = 2048 // grid_rows
    new_H = 1536 // grid_rows
    
    grid_image = np.ones((1536+border_px*(grid_rows-1), (2048//grid_rows)*grid_cols+border_px*(grid_cols-1), 3), dtype=np.uint8) * 255

    for idx, img_name in enumerate(name_list):
        img = image_dict[img_name].copy()
        cv2.putText(img, img_name, (80, 120), 1, 10, (255, 255, 0), 3)
        resized_img = cv2.resize(img, (new_W, new_H))
        
        r_idx = idx // grid_cols
        c_idx = idx % grid_cols

        r_start = r_idx * (new_H + border_px)
        c_start = c_idx * (new_W + border_px)
        grid_image[r_start:r_start+resized_img.shape[0], c_start:c_start+resized_img.shape[1]] = resized_img

    return grid_image