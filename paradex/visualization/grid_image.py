import cv2
import numpy as np

def grid_image(image_dict, cam_id_list=None):
    grid_image = np.zeros((1536, 2048, 3), dtype=np.uint8)
    grid_w = 6
    grid_h = 4
    if cam_id_list is None:
        cam_id_list = list(image_dict.keys())
        cam_id_list.sort()

    for idx, cam_id in enumerate(cam_id_list):
        if cam_id not in image_dict:
            continue
        img = image_dict[cam_id]
        img = cv2.resize(img, (2048//grid_w, 1536//grid_h))
        
        row = idx // grid_w
        col = idx % grid_w
        y_start = row * (1536//grid_h)
        y_end = (row + 1) * (1536//grid_h)
        x_start = col * (2048//grid_w)
        x_end = (col + 1) * (2048//grid_w)  

        grid_image[y_start:y_end, x_start:x_end] = img
    return grid_image