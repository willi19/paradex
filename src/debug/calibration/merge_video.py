import os
from paradex.utils.io import get_video_list, calib_path_list
from paradex.video.merge_video import merge_video
import cv2
import numpy as np

name = "20250322020521"
ind = "5"

for calib_path in calib_path_list:
    for name in os.listdir(calib_path):
        img_path = os.path.join(calib_path, name, ind, "images")
        kypt_path = os.path.join(calib_path, name, ind, "keypoints")

        index_list = os.listdir(img_path)
        index_list.sort(key=lambda x: int(x))
        for index in index_list:
            img_list = os.listdir(os.path.join(img_path, index))
            kypt_list = os.listdir(os.path.join(kypt_path, index))
            
            frame = []
            for img_name in img_list:
                name = img_name.split(".")[0]
                kypt_name = name + "_corners.npy"
                
                img = cv2.imread(os.path.join(img_path, index, img_name))
                kypt = np.load(os.path.join(kypt_path, index, kypt_name))
                kypt = np.array(kypt, dtype=np.int32)
                for k in kypt:                
                    cv2.circle(img, tuple(k), 5, (0, 255, 0), -1)
                frame.append(img)
            frame = np.hstack(frame)
            frame = cv2.resize(frame, (1920, 1080))
            cv2.imshow("frame", frame)
            cv2.waitKey(0)
