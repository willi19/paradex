import json
from paradex.utils.io import get_video_list
import argparse
import os
import matplotlib.pyplot as plt

homepath = os.path.expanduser('~/')
download_dir = os.path.join(homepath, 'download')

def main():
    parser = argparse.ArgumentParser(description='Create a timestamp file for a list of videos')
    parser.add_argument('--name', type=str, help='Path to a file containing a list of videos')
    args = parser.parse_args()

    
    index_list = os.listdir(os.path.join(download_dir, args.name))
    for index in ["1"]:#index_list:
        timestamp_list = {}
        for video in os.listdir(os.path.join(download_dir, args.name, index, 'videos')):
            if video.endswith('_timestamp.json'):
                serial_num = video.split('_')[0]
                with open(os.path.join(download_dir, args.name, index, 'videos', video), 'r') as f:
                    timestamp_list[serial_num] = json.load(f)

        for serial_num, ts_dict in timestamp_list.items():
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