import cv2
import os
from paradex.utils.io import home_dir
import numpy as np
import json
import matplotlib.pyplot as plt


index_list = ["0","1","2","3","4","5"]
for index in index_list:
    video_dir = os.path.join(home_dir,"captures1","calibration","20250323121043", index,"video")
    vid_list = os.listdir(os.path.join(video_dir))
    video_dir = os.path.join(home_dir,"captures2","calibration","20250323121043", index,"video")
    vid_list += os.listdir(os.path.join(video_dir))
    
    timestamp_list = []
    for vid_name in vid_list:
        if ".json" not in vid_name:
            continue
        ts_path = os.path.join(video_dir, vid_name)
        with open(ts_path) as f:
            ts = json.load(f)
        timestamp_list.append(ts)
    
        timestamp = np.array(ts["timestamps"])
        frame = np.array(ts["frameID"])

        frame_diff = frame[1:] - frame[:-1]
        timestamp_diff = timestamp[1:] - timestamp[:-1]
        fps = timestamp_diff / frame_diff
        print(len(frame_diff))
        for i, f in enumerate(frame_diff):
            if f == 2:
                print(f, timestamp_diff[i], fps[i])
        plt.hist(timestamp_diff,linestyle=None)
        plt.show()
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