import os
import argparse

import numpy as np
import json
import cv2
from dex_robot.utils.file_io import shared_path, load_camparam, load_c2r, download_path, rsc_path
from paradex.utils.marker import detect_aruco, triangulate, ransac_triangulation
import tqdm
from dex_robot.utils.robot_wrapper import RobotWrapper

def fill_framedrop(cam_timestamp):
    frameID = cam_timestamp["frameID"]
    pc_time = np.array(cam_timestamp["pc_time"])
    timestamp = np.array(cam_timestamp["timestamps"])

    time_delta = (1/30)# (pc_time[-1]-pc_time[0])/(frameID[-1]-frameID[0])
    offset = np.mean(pc_time - (np.array(frameID)-1)*time_delta)

    pc_time_nodrop = []
    frameID_nodrop = []

    for i in range(1, frameID[-1]+1):
        frameID_nodrop.append(i)
        pc_time_nodrop.append((i-1)*time_delta+offset)

    return pc_time_nodrop, frameID_nodrop

def load_timestamp(name):
    timestamp_list = []
    frame_num_list = []
    capture_ind = []

    index_list = os.listdir(os.path.join(shared_path, "capture", name))

    for index in index_list:
        timestamp = json.load(open(os.path.join(shared_path, "capture", name, index, "camera_timestamp.json")))
        pc_time, frameID = fill_framedrop(timestamp)

        selected_frame = json.load(open(os.path.join(shared_path, "capture", name, index, "selected_frame.json")))
        frame_num_list.append(len(selected_frame))

        
        for i in range(len(selected_frame)):
            timestamp_list.append([])
            s_f = selected_frame[str(i)]
            for (start, end) in s_f:
                for tmp in range(start-1, end):
                    timestamp_list[-1].append(pc_time[tmp])
                    if frameID[tmp] != tmp+1:
                        print("frameID error: ", frameID[tmp], tmp+1)
                
            timestamp_list[-1] = np.array(timestamp_list[-1])
            capture_ind.append(index)

    return timestamp_list, capture_ind

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args()

    name_list = [args.name] if args.name else os.listdir(os.path.join(download_path, 'processed'))
    
    robot = RobotWrapper(
        os.path.join(rsc_path, "xarm6", "xarm6_allegro_wrist_mounted_rotate.urdf")
    )
    link_index = robot.get_link_index("link5")
    finger_marker = {11:"ring_proximal", 13:"middle_proximal", 14:"index_proximal", 10:"thumb_proximal"}
    finger_index = {f"{finger_name}_proximal":robot.get_link_index(f"{finger_name}_proximal") for finger_name in ["thumb", "index", "middle", "ring"]}

    hand_marker_id_list = [10, 11, 13, 14]
    arm_marker_id_list = [262, 263, 264, 265, 266]

    for name in name_list:
        timestamp_list, capture_ind = load_timestamp(name)
        index_list = os.listdir(os.path.join(download_path, 'processed', name))
        capture_process_ind = []

        arm_timedelta = []
        hand_timedelta = []

        for index in index_list:
            root_path = os.path.join(download_path, 'processed', name, index)
            timestamp = timestamp_list[int(index)]
            c2r = load_c2r(root_path)

            marker_pose = np.load(os.path.join(root_path, "marker_pos.npy"), allow_pickle=True).item()            
            serial_list = [vid_name.split(".")[0] for vid_name in os.listdir(os.path.join(root_path, "video"))]

            seq_len = 0#len(os.listdir(os.path.join(root_path, "video_extracted", serial_list[0])))
            for serial_num in serial_list:
                img_list = [int(img_name.split(".")[0]) for img_name in os.listdir(os.path.join(root_path, "video_extracted", serial_num))]
                seq_len = max(seq_len, max(img_list))
                
            # if len(timestamp) != seq_len:
            #     print("timestamp error: ", len(timestamp), seq_len)
            #     break
            
            
            timestamp_camera = []
            for fid in tqdm.tqdm(range(seq_len)):
                marker_3d = np.load(os.path.join(root_path, "marker3d", f"{fid:05d}.npy"), allow_pickle=True).item()
                if len(marker_3d) == 0:
                    continue
                timestamp_camera.append({'time':timestamp[fid], 'marker_3d': marker_3d})

            
            robot_timestamp = np.load(os.path.join(shared_path, "capture", name, capture_ind[int(index)], "arm", "timestamp.npy"))
            robot_value = np.load(os.path.join(shared_path, "capture", name, capture_ind[int(index)], "arm", "state.npy"))
            
            hand_timestamp = np.load(os.path.join(shared_path, "capture", name, capture_ind[int(index)], "hand", "timestamp.npy"))
            hand_value = np.load(os.path.join(shared_path, "capture", name, capture_ind[int(index)], "hand", "state.npy"))

            time_delta = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
            min_time_delta_arm = -1
            min_time_delta_hand = -1
            min_err_arm = 1e10
            min_err_hand = 1e10

            for td in time_delta:
                ri = 0
                hi = 0

                err_r = 0
                cnt_r = 0

                err_h = 0
                cnt_h = 0
                
                for i in range(len(timestamp_camera)):
                    while ri < len(robot_timestamp)-1 and abs(robot_timestamp[ri] - timestamp_camera[i]["time"] + td) > abs(robot_timestamp[ri+1] - timestamp_camera[i]["time"] + td):
                        ri += 1
                    
                    while hi < len(hand_timestamp)-1 and abs(hand_timestamp[hi] - timestamp_camera[i]["time"] + td) > abs(hand_timestamp[hi+1] - timestamp_camera[i]["time"] + td):
                        hi += 1

                    qpos = np.zeros(22)

                    qpos[:6] = robot_value[ri]
                    qpos[6:] = hand_value[hi]

                    robot.compute_forward_kinematics(qpos)
                    link_pose = robot.get_link_pose(link_index)
                    for marker_id in arm_marker_id_list:
                        if marker_id not in timestamp_camera[i]["marker_3d"].keys():
                            continue
                        
                        cam_marker_pose = timestamp_camera[i]["marker_3d"][marker_id]
                        if cam_marker_pose is None:
                            continue
                        
                        robot_marker_pose = c2r @ link_pose @ marker_pose[marker_id]
                        robot_marker_pose = robot_marker_pose[:3, :] / robot_marker_pose[3, :]
                        robot_marker_pose = robot_marker_pose.T
                        
                        err_r += np.linalg.norm(cam_marker_pose - robot_marker_pose, axis=1).sum()
                        cnt_r += 1
                    
                    # for marker_id in hand_marker_id_list:
                    #     if marker_id not in timestamp_camera[i]["marker_3d"].keys():
                    #         continue
                        
                    #     cam_marker_pose = timestamp_camera[i]["marker_3d"][marker_id]
                    #     if cam_marker_pose is None:
                    #         continue
                        
                    #     finger_pose = robot.get_link_pose(finger_index[finger_marker[marker_id]])

                    #     finger_marker_pose = c2r @ finger_pose @ marker_pose[marker_id]
                    #     finger_marker_pose = finger_marker_pose[:3, :] / finger_marker_pose[3, :]
                    #     finger_marker_pose = finger_marker_pose.T
                        
                    #     err_h += np.linalg.norm(cam_marker_pose - finger_marker_pose, axis=1).sum()
                    #     cnt_h += 1

                # print("arm", "time delta: ", td, "error: ", err_r / cnt_r)
                # print("hand", "time delta: ", td, "error: ", err_h / cnt_h)

                if err_r / cnt_r < min_err_arm:
                    min_err_arm = err_r / cnt_r
                    min_time_delta_arm = td

                # if err_h / cnt_h < min_err_hand:
                #     min_err_hand = err_h / cnt_h
                #     min_time_delta_hand = td

            # print("min arm time delta: ", min_time_delta_arm, "error: ", min_err_arm)
            # print("min hand time delta: ", min_time_delta_hand, "error: ", min_err_hand)
            arm_timedelta.append(min_time_delta_arm)
            hand_timedelta.append(min_time_delta_hand)
        print("arm time delta: ", arm_timedelta)
        print("hand time delta: ", hand_timedelta)