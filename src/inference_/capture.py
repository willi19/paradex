import threading
import queue
import numpy as np
import cv2
import json
import time
import zmq
import os
import shutil
from paradex.utils.file_io import config_dir, shared_dir
from paradex.io.capture_pc.connect import git_pull, run_script

import math
from pathlib import Path
from scene import Scene
import torch
from geometry import get_visualhull_ctr

import time
import numpy as np
import pickle
import os
from paradex.utils.metric import get_pickplace_timing, compute_mesh_to_ground_distance
import open3d as o3d
from paradex.utils.file_io import rsc_path
from paradex.robot import RobotWrapper
from paradex.io.robot_controller import XArmController, AllegroController, InspireController
import transforms3d as t3d
from scipy.spatial.transform import Rotation as R
import chime

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"CUDA is available. Using device: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

BOARD_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 128, 255)
]
hide_list = ['22641005','22645021','23280594','23180202','22641023','23029839','22640993']


LINK62PALM = np.array(
    [
        [0, -1, 0, 0.0],
        [1, 0, 0, 0.0],#0.035],
        [0, 0, 1, -0.18],
        [0, 0, 0, 1],
    ]
)
demo_path = "data_Icra/teleoperation/bottle"
demo_path_list = os.listdir(demo_path)
demo_path_list.sort()

arm_name = "xarm"
hand_name = "allegro"

obj_mesh = o3d.io.read_triangle_mesh(os.path.join(rsc_path, "bottle", "bottle.obj"))

robot = RobotWrapper(
    os.path.join(rsc_path, "xarm6", "xarm6_allegro_wrist_mounted_rotate.urdf")
)
link_index = robot.get_link_index("palm_link")
lift_T = 100

def homo2cart(h):
    def project_to_so3(R):
        U, _, Vt = np.linalg.svd(R)
        R_proj = U @ Vt
        if np.linalg.det(R_proj) < 0:
            U[:, -1] *= -1
            R_proj = U @ Vt
        return R_proj


    if h.shape == (4, 4):
        t = h[:3, 3]
        R = h[:3, :3]
        
        R_proj = project_to_so3(R)

        axis, angle = t3d.axangles.mat2axangle(R_proj)
        axis_angle = axis * angle
    else:
        raise ValueError("Invalid input shape.")
    return np.concatenate([t, axis_angle])

def initialize_teleoperation(save_path):
    controller = {}
    if arm_name == "xarm":
        controller["arm"] = XArmController(save_path)

    if hand_name == "allegro":
        controller["hand"] = AllegroController(save_path)
        
    elif hand_name == "inspire":
        controller["hand"] = InspireController(save_path)
    
    return controller


def get_object_pose():
    f = open("data_Icra/teleoperation/bottle/1/obj_traj.pickle", "rb")
    obj_pose = pickle.load(f)["bottle"][0]
    
    obj_pose[:2, 3] = np.array([0.57176, 0.047841])  # Adjusted position
    f.close()
    return obj_pose

def determine_theta():
    return 0

def determine_traj_idx():
    return 1

def load_demo(demo_name):
    robot_prev = RobotWrapper(
        os.path.join(rsc_path, "xarm6", "xarm6_allegro_wrist_mounted_rotate_prev.urdf")
    )
    
    obj_T = pickle.load(open(os.path.join(demo_path, demo_name, "obj_traj.pickle"), "rb"))['bottle']
    robot_traj = np.load(os.path.join(demo_path, demo_name, "robot_qpos.npy"))
    target_traj = np.load(os.path.join(demo_path, demo_name, "target_qpos.npy"))

    T = obj_T.shape[0]

    height_list = []

    for step in range(T):
        h = compute_mesh_to_ground_distance(obj_T[step], obj_mesh)    
        height_list.append(h)

    end_T, _ = get_pickplace_timing(height_list)
    
    dist = 0.07
    for step in range(T):
        q_pose = robot_traj[step]
        robot_prev.compute_forward_kinematics(q_pose)
        wrist_pose = robot_prev.get_link_pose(link_index)
    
        obj_wrist_pose = np.linalg.inv(obj_T[step]) @ wrist_pose
        obj_pos = obj_wrist_pose[:2, 3]

        d = np.linalg.norm(obj_pos[:2])
        if d < dist:
            start_T = step
            break

    robot_pose = []
    hand_action = []
    obj_pose = obj_T[start_T].copy()
    tx, ty = obj_pose[:2, 3]

    for i in range(start_T, end_T + 1 + lift_T):
        if i > end_T:
            robot_pose.append(robot_pose[-1].copy())
            robot_pose[-1][2, 3] += 0.001
            hand_action.append(target_traj[end_T, 6:])
            continue

        robot_prev.compute_forward_kinematics(robot_traj[i])
        wrist_pose = robot_prev.get_link_pose(link_index)
        
        # wrist_axangle = target_traj[i, 3:6]
        # angle = np.linalg.norm(wrist_axangle)
        # if angle > 1e-6:
        #     wrist_axis = wrist_axangle / angle
        # else:
        #     wrist_axis = np.zeros(3)
            
        # wrist_rotmat = t3d.axangles.axangle2mat(wrist_axis, angle)

        # wrist_pose = np.zeros((4, 4))
        # wrist_pose[:3, :3] = wrist_rotmat
        # wrist_pose[:3, 3] = robot_traj[i][:3]
        # wrist_pose = np.linalg.inv(obj_T[0]) @ wrist_pose

        hand_action.append(target_traj[i, 6:])
        wrist_pose[:2, 3] -= np.array([tx, ty])  # Adjust position relative to object
        robot_pose.append(wrist_pose.copy())
    
    robot_pose = np.array(robot_pose)
    hand_action = np.array(hand_action)
    
    
    desired_x = np.array([0, 1, 0])
    current_x = robot_pose[-1, :3, 0][:2]  # x축 방향 (2D)
    theta = np.arctan2(desired_x[1], desired_x[0]) - np.arctan2(current_x[1], current_x[0])

    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1],
    ])

    obj_pose[:2, 3] -= np.array([tx, ty])
    # robot_pose[:, :2, 3] -= np.array([tx, ty])
    robot_pose[:, 2, 3] += (0.08 - robot_pose[0, 2, 3])

    robot_pose = np.einsum('ij, kjl -> kil', rot_mat, robot_pose)
    # import pdb; pdb.set_trace()
    return obj_pose, robot_pose, hand_action

def get_traj(tx, ty, theta, wrist_pose_demo):
    rot_mat = R.from_euler('z', theta, degrees=True).as_matrix()
    rot_T = np.eye(4)
    rot_T[:3, :3] = rot_mat

    wrist_pose = wrist_pose_demo.copy()
    wrist_pose = np.einsum('ij, kjl -> kil', rot_T, wrist_pose)
    wrist_pose[:, :2, 3] += np.array([tx, ty]) 

    return wrist_pose

def make_grid_img_inorder(cur_img_dict, height, width):
    
    big_img_width = width * 6
    big_img_height = height * 4
    
    canvas = np.zeros((big_img_height, big_img_width, 3), dtype=np.uint8)

    for serial_idx, serial_num in enumerate(cur_img_dict):
        row = serial_idx // 6
        col = serial_idx % 6
        x_start = col * width
        y_start = row * height
        
        img = cur_img_dict[serial_num]
        if img is not None:    
            canvas[y_start:y_start+height, x_start:x_start+width] = img
        
    return canvas


# argsparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, required=True)
parser.add_argument('--rgb_framerate', type=int, default=0, help='Framerate receiving RGB images from cameras, 0 for not receiving RGB images')
args = parser.parse_args()

output_dir = Path('/home/temp_id/paradex_processing/visualize/cache')
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
transl_dir = output_dir / 'transl'
transl_dir.mkdir(parents=True, exist_ok=True)
grid_dir = output_dir / 'grid'
grid_dir.mkdir(parents=True, exist_ok=True)
robot_dir = output_dir / 'robot'
robot_dir.mkdir(parents=True, exist_ok=True)

# make scene
obj_name = args.root_path.split("/")[-2]
mask_sub_dir = 'mask_hq/%s/%05d'%(obj_name, 0)
mask_root = Path(args.root_path)/mask_sub_dir
org_scene = Scene(root_path=Path(args.root_path), rescale_factor=1.0, mask_dir_nm=mask_sub_dir, device=device)

# === SETUP ===
pc_info = json.load(open(os.path.join(config_dir, "environment", "pc.json"), "r"))
serial_list = []
for pc in pc_info.keys():
    serial_list.extend(pc_info[pc]['cam_list'])

context = zmq.Context() # socket
socket_dict = {}
terminate_dict = {pc: False for pc in pc_info.keys()}
start_dict = {pc: False for pc in pc_info.keys()}

saved_corner_img = {serial_num :np.zeros((1536, 2048, 3), dtype=np.uint8) for serial_num in serial_list}
cur_state = {serial_num: (np.array([]), np.array([]), 0) for serial_num in serial_list}
capture_idx = 0
capture_state = {pc: False for pc in pc_info.keys()}

# filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())

detection_results = {}
cur_rgb = {cam_id: None for cam_id in serial_list}

def listen_socket(pc_name, socket):
    while True:
        msg = socket.recv_string()
        if msg == "terminated":
            terminate_dict[pc_name] = True
            print(f"[{pc_name}] Terminated.")
            break
        
        elif msg == "camera_ready":
            start_dict[pc_name] = True
            print(f"[{pc_name}] Camera ready.")
            continue
        
        elif msg == "camera_error":
            print(f"[{pc_name}] Camera error.")
            terminate_dict[pc_name] = True
            continue

        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            print(f"[{pc_name}] Non-JSON message: {msg}")
            continue
        
        serial_num = data["serial_num"]

        if data.get("type") == "demo": # Get yolo output
            detections_xyxy =  np.array(data["detections.xyxy"], dtype=float)
            detections_confidence = np.array(data["detections.confidence"], dtype=float)
            
            detections_mask = None # data["detections.mask"]
            detections_bbox_center = None
            
            if detections_xyxy.size > 0:
                detections_bbox_center = detections_xyxy[:, :2] + (detections_xyxy[:, 2:] - detections_xyxy[:, :2]) / 2
                bbox = detections_xyxy[0]
                # bbox mask.
                detections_mask = np.zeros((1, 1536, 2048), dtype=bool)
                detections_mask[0, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = True

            if data["frame"] not in detection_results:
                detection_results[data["frame"]] = {}
            
            detection_results[data["frame"]][serial_num] = {}
            detection_results[data["frame"]][serial_num]["mask"] = detections_mask
            detection_results[data["frame"]][serial_num]["xyxy"] = detections_xyxy
            detection_results[data["frame"]][serial_num]["confidence"] = detections_confidence
            detection_results[data["frame"]][serial_num]["frame_num"] = data["frame"]
            detection_results[data["frame"]][serial_num]["bbox_center"] = detections_bbox_center
            cur_rgb[serial_num] = np.array(data["resized_rgb"]) if args.rgb_framerate > 0 and data["frame"]%(args.rgb_framerate) == 0 and data["resized_rgb"]!='None' else None
            
            print(f"[{pc_name}] Received data for frame {data['frame']} from {serial_num}. Detections: {len(detections_xyxy)} Length: {len(detection_results[data['frame']])}")
            
        else:
            print(f"[{pc_name}] Unknown JSON type: {data.get('type')}")


def wait_for_cameras():
    while True:
        all_ready = True
        for pc_name, info in pc_info.items():
            if not start_dict[pc_name]:
                all_ready = False
                break
        if all_ready:
            print("All cameras are ready.")
            break
        time.sleep(0.1)


def main_ui_loop():
    curr_frame = 10
    cam_N = 10

    C2R = np.load(f"{shared_dir}/handeye_calibration/20250617_171318/0/C2R.npy")
    C2R = np.linalg.inv(C2R) # convert to camera coordinate system
    
    # for visualization
    import matplotlib.pyplot as plt
    plt.ion()  # interactive mode on
    fig, ax = plt.subplots()
    # imshow_o
    imshow_obj = ax.imshow(np.zeros((1536, 3072, 3)))  # Convert BGR to RGB
    # plt.pause(0.001)
    
    # object_pose = get_object_pose()
    tx, ty =  -1, -1# object_pose[:2, 3]

    theta = determine_theta()
    traj_idx = determine_traj_idx()
    traj_idx = str(traj_idx)
    
    obj_pose_demo, wrist_pose_demo, hand_pose = load_demo(traj_idx)

    transformed_traj = None# get_traj(tx, ty, theta, wrist_pose_demo)
    sensors = initialize_teleoperation(None)
    
    robot_idx = 0
    robot_init = False
    last_robot_time = -1
        
    while True:
        
        if tx != -1 and last_robot_time == -1:            
            if hand_name is not None:
                sensors["hand"].set_homepose(hand_pose[0])
                sensors["hand"].home_robot()

            if arm_name is not None:
                sensors["arm"].set_homepose(homo2cart(transformed_traj[0] @ LINK62PALM))
                sensors["arm"].home_robot()

                home_start_time = time.time()
                while sensors["arm"].ready_array[0] != 1:
                    if time.time() - home_start_time > 0.3:
                        chime.warning()
                        home_start_time = time.time()
                    time.sleep(0.0008)
                chime.success()
            
            robot_init = True
            last_robot_time = time.time()
        
        if last_robot_time != -1 and time.time() - last_robot_time > 0.1 and robot_idx < len(transformed_traj):
            l6_pose = transformed_traj[robot_idx] @ LINK62PALM
            arm_action = homo2cart(l6_pose)
            hand_action = hand_pose[robot_idx]

            if arm_name is not None:                
                sensors["arm"].set_target_action(
                                arm_action
                        )
                
                arm_state = sensors["arm"].arm_state_array.copy()
                print(arm_state)
                np.save(os.path.join(robot_dir, "arm_state.npy"), arm_state)
                
            if hand_name is not None:
                sensors["hand"].set_target_action(
                                hand_action
                            )
                hand_state = sensors["hand"].hand_state_array.copy()
                np.save(os.path.join(robot_dir, "hand_state.npy"), hand_state)
            robot_idx += 1
            last_robot_time = time.time()
            print("robot move ", robot_idx)
                
                        
            
        
        if last_robot_time != -1 and curr_frame == robot_idx and robot_idx == len(transformed_traj):
            break
            
        if curr_frame in detection_results and len(detection_results[curr_frame]) < 22: 
            time.sleep(0.01)
            continue
        
        if curr_frame in detection_results and len(detection_results[curr_frame]) >= 22:
            detect_img = 0
            for cam_id in detection_results[curr_frame]:
                if detection_results[curr_frame][cam_id]["xyxy"].size > 0:
                    detect_img += 1
            if detect_img < cam_N:
                print("Not enough detections, waiting for more...")
                time.sleep(0.01)
                curr_frame += 1
                continue
            
            detection_results_curr = detection_results[curr_frame]
            confidence_dict = {cam_id: detection_results_curr[cam_id]["confidence"][0] for cam_id in detection_results_curr if detection_results_curr[cam_id]["confidence"].size > 0 and detection_results_curr[cam_id]["confidence"][0] > 0.001 and cam_id not in hide_list}
            # print(confidence_dict)
            top_n_cams2confidence = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)[:cam_N]
            top_n_cams = [cam_id for cam_id, confidence in top_n_cams2confidence]

            A = []
            for cam_id in top_n_cams:
                cx, cy = detection_results_curr[cam_id]["bbox_center"][0]
                P = org_scene.proj_matrix[cam_id]
                A.append(cx * P[2] - P[0])
                A.append(cy * P[2] - P[1])
            A = np.stack(A, axis=0)
            _, _, Vt = np.linalg.svd(A)
            X_h = Vt[-1]
            X = X_h[:3] / X_h[3]  # Convert from homogeneous coordinates to 3D point
            
            initial_translate = X @ C2R[:3, :3].T + C2R[:3, 3] # convert to camera coordinate system
            if tx == -1:
                tx = initial_translate[0]
                ty = initial_translate[1] - 0.02
                transformed_traj = get_traj(tx, ty, theta, wrist_pose_demo)
            
            print(f"{initial_translate}")
            np.save(os.path.join(transl_dir, f'transl_{curr_frame}.npy'), initial_translate)
            np.save(os.path.join(transl_dir, f'transl.npy'), initial_translate)
            
            if cur_rgb[serial_list[0]] is not None:
                grid_img = make_grid_img_inorder(cur_rgb, int(1536/16), int(2048/16))
                cv2.imwrite(os.path.join(grid_dir, f"grid_{curr_frame}.jpg"), grid_img)    
            curr_frame += 1
            time.sleep(0.01)
    
    for key in sensors.keys():
        sensors[key].quit()   

# Git pull and client run
pc_list = list(pc_info.keys())
git_pull("merging", pc_list)

run_script(f"python src/inference_/yolo.py --rgb_framerate {args.rgb_framerate}", pc_list)


# Connect pc with 5566 TCP
for pc_name, info in pc_info.items():
    ip = info["ip"]
    sock = context.socket(zmq.DEALER)
    sock.identity = b"server"
    sock.connect(f"tcp://{ip}:5556")
    socket_dict[pc_name] = sock

# Check socker registered
for pc_name, info in pc_info.items():
    socket_dict[pc_name].send_string("register")
    if socket_dict[pc_name].recv_string() == "registered":
        print(f"[{pc_name}] Registered.")
    
    # socket_dict[pc_name].send_string("filename:" + filename)

# Start per-socket listener
for pc_name, sock in socket_dict.items():
    threading.Thread(target=listen_socket, args=(pc_name, sock), daemon=True).start()

wait_for_cameras()
print("All cameras are ready.")
# Main UI loop
# while True:
#     time.sleep(0.1)  # Prevent busy-waiting
main_ui_loop()

# except Exception as e:
#     print(e)
#     for pc_name, sock in socket_dict.items():
#         sock.send_string("quit")
#         sock.close()

for pc_name, sock in socket_dict.items():
    sock.send_string("quit")
    sock.close()
        