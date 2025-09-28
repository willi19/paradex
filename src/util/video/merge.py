import argparse
from multiprocessing import Pool
import os

from paradex.image.merge import merge_image
from paradex.video.process_video import process_video_list

from paradex.video.convert_codec import change_to_h264

def load_info(video_dir):
    return None

def process_frame(img_dict, video_path, fid, data):
    frame = merge_image(img_dict)
    return frame

def process_frame_list(img_dict, video_dir, fid):
    frame = merge_image(img_dict)
    return frame

# td = 0.09 # latency difference between camera and sensor
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--out_path', type=str, default=None)
    
    args = parser.parse_args()

    dir_name = os.path.dirname(args.out_path)
    print(dir_name)
    os.makedirs(dir_name, exist_ok=True)
    
    process_video_list(args.path, 
            args.out_path, 
            None, 
            process_frame)
    change_to_h264(f"{args.out_path.split('.')[0]}_tmp.avi", args.out_path)