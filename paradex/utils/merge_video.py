import os
import json

import cv2
import numpy as np

from paradex.utils.io import get_video_list
import tqdm

cam_index = json.load(open('config/camera_index.json'))

def merge_video(input_dir, output_file):
    # Get list of AVI files in the directory
    video_list = get_video_list(input_dir)

    caps = {}
    timestamp_list = {}
    index_position = {}
    # Open video captures
    for f in video_list:
        serial_num = os.path.basename(f[0]).split("_")[0]
        if serial_num not in cam_index.keys():
            print(f"Camera index not found for {serial_num}")
            return
        caps[serial_num] = cv2.VideoCapture(f[0])
        timestamp_list[serial_num] = json.load(open(f[1],"r"))["frameID"]
        index_position[serial_num] = 0

    # Get frame properties from the first video
    first_key = list(caps.keys())[0]
    frame_width = int(caps[first_key].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[first_key].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(caps[first_key].get(cv2.CAP_PROP_FPS))
    
    # Get the maximum timestamp for video length
    max_frames = max([max(timestamps) for timestamps in timestamp_list.values()])

    rows, cols = 4, 6
    # Define output video size
    grid_width = frame_width // cols
    grid_height = frame_height // rows


    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    for frame_idx in tqdm.tqdm(range(max_frames)):
        merged_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # Fill with black if no frame
        # Read frames from each video only if the frame index is in timestamp_list
        for serial_num, cap in caps.items():
            offset_x = ((cam_index[serial_num]-1) % cols) * grid_width
            offset_y = (cam_index[serial_num]-1) // cols * grid_height
            
            if index_position[serial_num] == len(timestamp_list[serial_num]):
                continue
            
            if frame_idx == timestamp_list[serial_num][index_position[serial_num]]:
                ret, frame = cap.read()
                index_position[serial_num] += 1

                if ret:
                    frame = cv2.resize(frame, (grid_width, grid_height))
                    merged_frame[offset_y:offset_y + grid_height, offset_x:offset_x + grid_width] = frame
            
            cv2.putText(merged_frame, serial_num, (offset_x, offset_y+grid_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # merged_frame_tmp = cv2.resize(merged_frame, (frame_width*2//3, frame_height*2//3))
        # cv2.imshow("Merged Video", merged_frame_tmp)
        # cv2.waitKey(0)
        # Write to output video
        out.write(merged_frame)

    # Release resources
    for cap in caps.values():
        cap.release()
    out.release()

    print("Video processing complete. Output saved as", output_file)

def merge_video_synced(input_dir, output_file):
    # Get list of AVI files in the directory
    video_list = get_video_list(input_dir)

    caps = {}
    timestamp_list = {}
    index_position = {}
    # Open video captures
    for f in video_list:
        serial_num = os.path.basename(f[0]).split(".")[0]
        if serial_num not in cam_index.keys():
            print(f"Camera index not found for {serial_num}")
            return
        caps[serial_num] = cv2.VideoCapture(f[0])
        
    # Get frame properties from the first video
    first_key = list(caps.keys())[0]
    frame_width = int(caps[first_key].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[first_key].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(caps[first_key].get(cv2.CAP_PROP_FPS))
    
    # Get the maximum timestamp for video length
    rows, cols = 4, 6
    # Define output video size
    grid_width = frame_width // cols
    grid_height = frame_height // rows


    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    while True:
        merged_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # Fill with black if no frame
        # Read frames from each video only if the frame index is in timestamp_list
        all_down = True
        for serial_num, cap in caps.items():
            offset_x = ((cam_index[serial_num]-1) % cols) * grid_width
            offset_y = (cam_index[serial_num]-1) // cols * grid_height
            
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (grid_width, grid_height))
                merged_frame[offset_y:offset_y + grid_height, offset_x:offset_x + grid_width] = frame
                all_down = False            
            
        # merged_frame_tmp = cv2.resize(merged_frame, (frame_width*2//3, frame_height*2//3))
        # cv2.imshow("Merged Video", merged_frame_tmp)
        # cv2.waitKey(0)
        # Write to output video
        out.write(merged_frame)
        if all_down:
            break

    # Release resources
    for cap in caps.values():
        cap.release()
    out.release()

    print("Video processing complete. Output saved as", output_file)
