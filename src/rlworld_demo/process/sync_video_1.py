# Grasping timing
# multiview : 445
# iphone2 : 435

# start in multiview : 367 = 445 - (435-200) / 3
# start in iphone1 : 345
# start in iphone2 : 200

# end in iphone1 : 345 + (19204-200)*2 = 38349
# end in iphone2 : 19204
# end in multiview : 445 + (19202-435) / 3 = 6701

vid_range = {"iphone1.mov": (345, 38354), "iphone2.mov": (200, 19204), "multiview": (367, 6701)}

import os
import cv2
import numpy as np
import tqdm
from multiprocessing import Pool, cpu_count


from paradex.utils.file_io import shared_dir
from paradex.video.convert_codec import change_to_h264

def process_video(video_path, save_path, start=0, end=-1):
    cap = cv2.VideoCapture(video_path)

    if "multiview" in save_path:
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (2048, 1536))
        print(video_path)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    
    for i in tqdm.tqdm(range(num_frame), desc=f"Processing {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            error = True
            break

        if i >= start and i <= end:
            if np.sum(frame[:200, :200]) < 10 and prev_frame is not None:
                out.write(prev_frame)
            else:
                out.write(frame)

        if np.sum(frame) > 10:
            prev_frame = frame

    cap.release()
    out.release()

    # change_to_h264(save_path, save_path[:-8] + ".mp4")

def process_single_video_task(args):
    """단일 비디오 처리 태스크 (멀티프로세싱용)"""
    video_path, save_path, start, end = args
    
    try:
        process_video(video_path, save_path, start, end)
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return
    
for index in range(1, 2):
    tasks = []
    # multiview
    save_dir = os.path.join("RLWRLD_DEMO_FINAL", str(index))
    
    # multi_video_dir = os.path.join("RLWRLD_DEMO", f"{index}",  f"multiview")
    # for video_name in os.listdir(multi_video_dir):
    #     if video_name == "cam_param":
    #         continue
    #     start = vid_range["multiview"][0]
    #     end = vid_range["multiview"][1]
    #     video_path = os.path.join(multi_video_dir, video_name)
    #     os.makedirs(os.path.join(save_dir, "multiview"), exist_ok=True)
    #     save_path = os.path.join(save_dir, "multiview", f"{video_name[:-4]}_tmp.mp4")
    #     tasks.append((video_path, save_path, start, end))
        
    vid_name_list = ["iphone1.mov"]
    for vid_name in vid_name_list:
        video_path = os.path.join("RLWRLD_DEMO", str(index), vid_name)
        save_path = os.path.join(save_dir, f"{vid_name[:-4]}_tmp.mp4")
        start = vid_range[vid_name][0]
        end = vid_range[vid_name][1]
        tasks.append((video_path, save_path, start, end))
        # process_video(video_path, save_path, start, end)
        
    num_process = cpu_count() - 2
    with Pool(num_process) as p:
        p.map(process_single_video_task, tasks)

        