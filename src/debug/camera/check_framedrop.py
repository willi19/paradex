import matplotlib.pyplot as plt
import numpy as np
import os
import json
from paradex.utils.io import download_dir
import argparse
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot framedrop data.")
    parser.add_argument("--name", help="Path to the framedrop file.", default=None)

    args = parser.parse_args()
    obj_list = []
    if args.name == None:
        obj_list = os.listdir(os.path.join(download_dir, "capture"))
    else:
        obj_list.append(args.name)

    for obj_name in obj_list:
        capture_path = os.path.join(download_dir, "capture", obj_name)

        index_list = os.listdir(capture_path)
        for index in index_list:
            timestamp_list = glob.glob(os.path.join(capture_path, index, "videos", "*_timestamp.json"))
            if len(timestamp_list) != 24:
                print(index, len(timestamp_list),os.path.join(capture_path, index, "videos", "*_timestamp.json"))

            frame_ids = {}
            max_len = 0
            serial_ind = {}

            for i, timestamp in enumerate(timestamp_list):
                serial_num = os.path.basename(timestamp).split("_")[0]
                if serial_num == "camera":
                    serial_ind[serial_num] = "camera"
                else:
                    serial_ind[serial_num] = str(serial_num)
                with open(timestamp, 'r') as f:
                    ts_dict = json.load(f)

                frame_ids[serial_num] = np.array(ts_dict['frameID'])
                timestamp = np.array(ts_dict['timestamps'])

                fps = (timestamp[1:] - timestamp[:-1]) / (frame_ids[serial_num][1:] - frame_ids[serial_num][:-1])
                fps_min = np.min(fps)
                fps_max = np.max(fps)
                # print(serial_num, fps_min, fps_max)
                max_len = max(max_len, len(frame_ids[serial_num]))
            
            # for serial_num, fid in frame_ids.items():
            #     if len(fid) < max_len:
            #         for i in range(1, max_len+1):
            #             if i not in frame_ids[serial_num]:
            #                 print(serial_num, i)
            # for serial_num, frame_id in frame_ids.items():
            #     dropped_frames = [i for i in range(1, max_len+1) if i not in frame_id]
            #     print(serial_num, dropped_frames)
            #     plt.plot(dropped_frames, [serial_ind[serial_num]]*len(dropped_frames), 'o', label=f"Camera {serial_num}")
            # plt.savefig(os.path.join("image", obj_name, str(index), "framedrop.png"))
            # plt.show()
            # plt.clf()