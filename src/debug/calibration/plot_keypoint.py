import cv2
import os
from paradex.utils.io import home_path
import numpy as np



index_list = ["0","1","2"]
for index in index_list:
    image_dir = os.path.join(home_path,"captures1","calibration","20250323121043", index,"images")
    img_list = os.listdir(os.path.join(image_dir))
    img_list.sort()
    for i in img_list:
        image_path = os.path.join(image_dir, i, "23263780.jpg")
        
        if not os.path.exists(image_path):
            continue
        kypt_pth = image_path.replace("images","keypoints")
        kypt_pth = kypt_pth[:-4]+"_corners.npy"

        kypt = np.load(kypt_pth)
        img = cv2.imread(image_path)

        for k in kypt:
            cv2.circle(img, (int(k[0]),int(k[1])), 2, (0,0,255), -1)
        cv2.imshow("img",img)
        cv2.waitKey(0)
        
