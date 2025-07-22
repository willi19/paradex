import json
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import shutil

from paradex.io.contact.process import process_contact
from paradex.robot import RobotWrapper
from paradex.utils.file_io import rsc_path, shared_dir
from paradex.video.raw_video import fill_framedrop, get_synced_data
from paradex.geometry.coordinate import DEVICE2WRIST

# td = 0.09 # latency difference between camera and sensor
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_name', nargs="+", type=str, default=None)
    parser.add_argument('--grasp_type', nargs="+", type=str, default=None)
    args = parser.parse_args()

    process_list = []
    
    if args.obj_name == None:
        name_list = os.listdir(os.path.join(shared_dir, 'capture_', "lookup"))
        name_list.sort()

    else:
        name_list = [args.obj_name]
        
    for name in name_list:
        grasp_list = os.listdir(os.path.join(shared_dir, "capture_", "lookup", name))
        if args.grasp_type is not None and args.grasp_type in grasp_list:
            process_list.append((name, args.grasp_type))
        if args.grasp_type is None:
            for grasp_name in grasp_list:
                process_list.append((name, grasp_name))
    
    for name, grasp_type in process_list:
        root_dir = os.path.join(shared_dir, "capture_", "lookup", name, grasp_type)
        index_list = os.listdir(root_dir)
        
        for index in index_list:
            index_dir = os.path.join(os.path.join(root_dir, str(index)))
            raw_dir = os.path.join(os.path.join(index_dir, "raw"))
            
            valid = True
            
            arm_name = None
            for tmp_name in ['xarm', 'franka']:
                if tmp_name in os.listdir(raw_dir):
                    arm_name = tmp_name
                    break
                            
            hand_name = None
            for tmp_name in ['inspire', 'allegro']:
                if tmp_name in os.listdir(raw_dir):
                    hand_name = tmp_name
                    break
            
            if arm_name == None or hand_name == None:
                # shutil.rmtree(index_dir)
                valid = False
            
            print(arm_name, hand_name, name, grasp_type, index)
            for data_name in ["state", "timestamp"]:#, "C2R.npy"]:
                if data_name not in os.listdir(raw_dir):
                    valid = False
            
            if "cam_param" not in os.listdir(index_dir):
                valid = False
                
            if not valid:
                print(f"{name} {grasp_type} invalid")
                continue
                
            hand2last_link = np.linalg.inv(DEVICE2WRIST[arm_name]) @ DEVICE2WRIST[hand_name]
            timestamp = json.load(open(os.path.join(raw_dir, "timestamp", "camera_timestamp.json")))
            pc_time, fid = fill_framedrop(timestamp)
            os.makedirs(os.path.join(index_dir), exist_ok=True)
            
            # process arm
            arm_action_orig = np.load(os.path.join(raw_dir, arm_name, "action.npy")) # T X 4 X 4
            arm_qpos_orig = np.load(os.path.join(raw_dir, arm_name, "position.npy")) # T X dof
            arm_pc_time = np.load(os.path.join(raw_dir, arm_name, "time.npy"))
            
            arm_action_sync = get_synced_data(pc_time, arm_action_orig, arm_pc_time)
            arm_qpos_sync = get_synced_data(pc_time, arm_qpos_orig, arm_pc_time)
            os.makedirs(os.path.join(index_dir, arm_name), exist_ok=True)
            np.save(os.path.join(index_dir, arm_name, "action.npy"), arm_action_sync)
            np.save(os.path.join(index_dir, arm_name, "qpos.npy"), arm_qpos_sync)
                      
            hand_action_orig = np.load(os.path.join(raw_dir, hand_name, "action.npy"))
            hand_qpos_orig = np.load(os.path.join(raw_dir, hand_name, "position.npy"))
            hand_pc_time = np.load(os.path.join(raw_dir, hand_name, "time.npy"))
            
            hand_action_sync = get_synced_data(pc_time, hand_action_orig, hand_pc_time)
            hand_qpos_sync = get_synced_data(pc_time, hand_qpos_orig, hand_pc_time)
            
            robot = RobotWrapper(
                os.path.join(rsc_path, f"robot/{arm_name}/{arm_name}.urdf")
            )
            end_link = robot.get_link_index(robot.get_end_links()[0])
            
            hand_pose = []
            for qpos in arm_qpos_sync:
                robot.compute_forward_kinematics(qpos)
                last_link_pose = robot.get_link_pose(end_link)
                hand_pose.append(last_link_pose @ hand2last_link)
            hand_pose = np.array(hand_pose)
            
            os.makedirs(os.path.join(index_dir, hand_name), exist_ok=True)
            np.save(os.path.join(index_dir, hand_name, "action.npy"), hand_action_sync)
            np.save(os.path.join(index_dir, hand_name, "qpos.npy"), hand_qpos_sync)
            np.save(os.path.join(index_dir, hand_name, "position.npy"), hand_pose)
            
            state_orig = np.load(os.path.join(raw_dir, "state", "state.npy"))
            state_pc_time = np.load(os.path.join(raw_dir, "state", "time.npy"))
            state_sync = get_synced_data(pc_time, state_orig, state_pc_time)
            
            np.save(os.path.join(index_dir, "state.npy"), state_sync)

            # print(state_sync.shape, hand_action_sync.shape, hand_qpos_sync.shape, hand_pose.shape, arm_action_sync.shape, arm_qpos_sync.shape)