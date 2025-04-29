import cv2
import os
from paradex.utils.io import home_dir
import numpy as np



index_list = ["4"]
for index in index_list:
    video_dir = os.path.join(home_dir,"captures1","calibration","20250323121043", index,"video")
    vid_list = os.listdir(os.path.join(video_dir))
    cap_list = []
    for vid_name in vid_list:
        if ".avi" not in vid_name:
            continue
        vid_path = os.path.join(video_dir, vid_name)
        cap = cv2.VideoCapture(vid_path)
        cap_list.append(cap)
    # while True:
    #     img = np.zeros((960,2560,3), np.uint8)
    #     exist = True
    #     for i, cap in enumerate(cap_list):
    #         ret, frame = cap.read()
    #         if not ret:
    #             exist = False
    #             break
    #         frame = cv2.resize(frame, (1280,960))
    #         img[:,i*1280:(i+1)*1280] = frame
    #     cv2.imshow("img",img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break