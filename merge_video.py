from paradex.video.convert_codec import change_to_h264
from paradex.video.process_video import process_video_list
from paradex.image.merge import merge_image
from paradex.utils.file_io import shared_dir

import cv2
import os

def process_frame(img_dict, video_dir, fid, data):
    frame = merge_image(img_dict)
    frame = cv2.resize(frame, (2048, 1536))
    return frame

process_video_list(os.path.join(shared_dir, "inference_", "lookup", "pringles", "stand_free", "13", "videos"), "fail_merged.mp4", None, process_frame)
change_to_h264("data_merged_tmp.avi", "fail_merged.mp4")