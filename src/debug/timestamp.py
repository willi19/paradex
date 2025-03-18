import json
from paradex.utils.io import get_video_list
import argparse
import os
import matplotlib.pyplot as plt

homepath = os.path.expanduser('~/')
capture_path = [os.path.join(homepath, f"captures{i}") for i in range(1, 3)]

def main():
    parser = argparse.ArgumentParser(description='Create a timestamp file for a list of videos')
    parser.add_argument('--name', type=str, help='Path to a file containing a list of videos')
    args = parser.parse_args()

    timestamp_list = []
    for path in capture_path:
        timestamp_path_list = {os.path.basename(f[0]):f[1] for f in get_video_list(os.path.join(path, args.name))}
        for serial_num, timestamp_path in timestamp_path_list.items():
            with open(timestamp_path, 'r') as f:
                ts_dict = json.load(f)
            timestamp_list.append((serial_num, ts_dict))

    for serial_num, ts_dict in timestamp_list:
        frame_ids = ts_dict['frameID']
        timestamps = ts_dict['timestamps']
        timestamps = [(timestamps[i] - timestamps[0])/(10**8) for i in range(len(timestamps))]
        print(f"Camera {serial_num} has {len(frame_ids)} frames {timestamps[-1]} seconds apart")
        print("camera", serial_num, "fps:", frame_ids[-1] / timestamps[-1])
        plt.plot(frame_ids, timestamps, label=f"Camera {serial_num}")
        

    plt.title("Frame ID vs. Adjusted Timestamps")
    plt.show()
    plt.legend()

if __name__ == '__main__':
    main()