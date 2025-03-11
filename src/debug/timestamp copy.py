import json
from paradex.utils.io import get_video_list
import argparse
import os
import matplotlib.pyplot as plt

homepath = os.path.expanduser('~/')
def main():
    parser = argparse.ArgumentParser(description='Create a timestamp file for a list of videos')
    parser.add_argument('--name', type=str, help='Path to a file containing a list of videos')
    args = parser.parse_args()

    video_list = get_video_list(os.path.join(homepath, "download", args.name))

    for (video_path, ts_path) in video_list:
        serial_num = os.path.basename(video_path).split('_')[0]
        with open(ts_path, 'r') as f:
            ts_dict = json.load(f)

        frame_ids = ts_dict['frameID']
        timestamps = ts_dict['timestamps']
        timestamps = [timestamps[i] - timestamps[0] for i in range(len(timestamps))]
        print(f"Camera {serial_num} has {len(frame_ids)} frames {timestamps[-1]} seconds apart")
        # plt.plot(frame_ids, timestamps, label=f"Camera {serial_num}")
        
    # plt.xlabel("Frame ID")
    # plt.ylabel("Timestamp Difference (s)")
    # plt.title("Frame ID vs. Adjusted Timestamps")
    # plt.show()
    # plt.legend()

if __name__ == '__main__':
    main()