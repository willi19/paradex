import threading
import queue
import numpy as np
import cv2
import json
import time
import zmq
import os
from paradex.utils.file_io import config_dir, shared_dir
from paradex.io.capture_pc.connect import git_pull, run_script
import math
import torch

from pathlib import Path
from scene import Scene
from yolo_world_module import YOLO_MODULE

# === SETUP ===
pc_info = json.load(open(os.path.join(config_dir, "environment", "pc.json"), "r"))
serial_list = []
for pc in pc_info.keys():
    serial_list.extend(pc_info[pc]['cam_list'])

context = zmq.Context()
socket_dict = {}
terminate_dict = {pc: False for pc in pc_info.keys()}
start_dict = {pc: False for pc in pc_info.keys()}

saved_corner_img = {serial_num:np.zeros((1536, 2048, 3), dtype=np.uint8) for serial_num in serial_list}
cur_state = {serial_num:(np.array([]), np.array([]), 0) for serial_num in serial_list}
capture_idx = 0
capture_state = {pc: False for pc in pc_info.keys()}

filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())

# def draw_charuco_corners_custom(image, corners, color=(0, 255, 255), radius=4, thickness=2, ids=None):
#     for i in range(len(corners)):
#         corner = tuple(int(x) for x in corners[i][0])
#         cv2.circle(image, corner, radius, color, thickness)
#         if ids is not None:
#             cv2.putText(image, str(int(ids[i])), (corner[0] + 5, corner[1] - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, lineType=cv2.LINE_AA)

asdf = {}

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

        elif msg == "save_finish":
            capture_state[pc_name] = False
            print(f"[{pc_name}] Save finished.")
            continue

        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            print(f"[{pc_name}] Non-JSON message: {msg}")
            continue
        
        serial_num = data["serial_num"]
        if data.get("type") == "charuco":
            # corners = np.array(result["checkerCorner"], dtype=np.float32)
            # ids = np.array(result["checkerIDs"], dtype=np.int32).reshape(-1, 1)
            frame = data["frame"]
            asdf[serial_num] = frame
            # cur_state[serial_num] = (corners, ids, frame)

            # if result["save"]:
            #     draw_charuco_corners_custom(saved_corner_img[serial_num], corners, BOARD_COLORS[2], 5, -1, ids)

        else:
            print(f"[{pc_name}] Unknown JSON type: {data.get('type')}")

def main_loop(yolo_module):
    current_idx = 1
    while True:
        os.makedirs(os.path.join(shared_dir, "demo_250618", "pringles", str(current_idx), "images_undistorted"), exist_ok=True)
        cur_cnt = 0
        
        for serial_num in serial_list:
            if serial_num in asdf:
                if asdf[serial_num] >= current_idx:
                    current_idx = asdf[serial_num]
                    cur_cnt += 1
        print(asdf, cur_cnt, current_idx)
        if cur_cnt == len(serial_list):
            mask_sub_dir = 'mask_hq/%s/%05d'%("pringles", 0)
            device = torch.device("cuda:0")
            org_scene = Scene(root_path=Path(root_path), rescale_factor=1.0, mask_dir_nm=mask_sub_dir, device=device)

            # DO YOLO
            detection_results = {}
            results_img = []
            for cam_id in org_scene.cam_ids:

                rgb_img = org_scene.get_image_demo(cam_id, current_idx)
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                detections = yolo_module.process_img(rgb_img, with_segmentation=False)
                # output_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                # res_vis = yolo_module.annotate_image(output_image, detections, categories=yolo_module.categories, with_confidence=True)
                # imgs.append(res_vis)
                if detections.xyxy.size > 0:
                    detections.bbox_center = detections.xyxy[:, :2] + (detections.xyxy[:, 2:] - detections.xyxy[:, :2]) / 2
                    detections.mask = np.zeros((1, rgb_img.shape[0], rgb_img.shape[1]), dtype=bool)
                    detections.mask[:, int(detections.xyxy[0, 1]):int(detections.xyxy[0, 3]), int(detections.xyxy[0, 0]):int(detections.xyxy[0, 2])] = True
                detection_results[cam_id] = detections
                canvas = yolo_module.annotate_image(rgb_img, detections, categories=yolo_module.categories, with_confidence=True)
                canvas = putText(canvas, cam_id, color=(0,0, 255))
                # print(type(canvas))
                results_img.append(canvas[::4, ::4])
            # plot grid image
            cv2.imwrite(f'debug_grid_{current_idx}.png',make_grid_image_np(np.array(results_img), 4,6))

            confidence_dict = {cam_id: detection_results[cam_id].confidence.item() for cam_id in detection_results if detection_results[cam_id].confidence.size > 0 and detection_results[cam_id].confidence > 0.001 and cam_id not in hide_list}
            cam_N = 10
            top_n_cams2confidence = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)[:cam_N]
            top_n_cams = [cam_id for cam_id, confidence in top_n_cams2confidence]
            
            # triangulation_start = time.time()
            A = []
            for cam_id in top_n_cams:
                cx, cy = detection_results[cam_id].bbox_center[0]
                P = org_scene.proj_matrix[cam_id]
                A.append(cx * P[2] - P[0])
                A.append(cy * P[2] - P[1])
            A = np.stack(A, axis=0)
            _, _, Vt = np.linalg.svd(A)
            X_h = Vt[-1]
            X = X_h[:3] / X_h[3]  # Convert from homogeneous coordinates to 3D point
            
            object_output_dir = Path(root_path)/(f'{obj_name}_optim')
            os.makedirs(object_output_dir, exist_ok=True)

            json.dump(vars(args), open(object_output_dir/'params.json','w'))

            object_final_output_dir = object_output_dir/'final'
            os.makedirs(object_final_output_dir, exist_ok=True)
            
            C2R = np.load(f"{shared_dir}/handeye_calibration/20250617_171318/0/C2R.npy")
            C2R = np.linalg.inv(C2R) # convert to camera coordinate system
            # print(C2R)
            initial_translate = X @ C2R[:3, :3].T + C2R[:3, 3] # convert to camera coordinate system
            print(initial_translate)
            
            result_dict = {'t':initial_translate}
            import pickle
            pickle.dump(result_dict, open(object_final_output_dir/f'init_transl.pickle','wb'))
                
            current_idx += 1

# def main_ui_loop():
#     num_images = len(serial_list)
#     grid_cols = math.ceil(math.sqrt(num_images))
#     grid_rows = math.ceil(num_images / grid_cols)
#     border_px = 20

#     new_W = 2048 // grid_rows
#     new_H = 1536 // grid_rows

#     while True:
#         all_disconnected = True
#         for pc_name, terminated in terminate_dict.items():
#             if not terminated:
#                 all_disconnected = False
#         if all_disconnected:
#             break
        
#         grid_image = np.ones((1536+border_px*(grid_rows-1), (2048//grid_rows)*grid_cols+border_px*(grid_cols-1), 3), dtype=np.uint8) * 255
#         for idx, serial_num in enumerate(serial_list):
#             img = saved_corner_img[serial_num].copy()
#             corners, ids, frame = cur_state[serial_num]
#             if corners.shape[0] > 0:
#                 draw_charuco_corners_custom(img, corners, BOARD_COLORS[1], 5, -1, ids)
#             img = cv2.putText(img, f"{serial_num} {frame}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 0), 12)

#             resized_img = cv2.resize(img, (new_W, new_H))
            
#             r_idx = idx // grid_cols
#             c_idx = idx % grid_cols

#             r_start = r_idx * (new_H + border_px)
#             c_start = c_idx * (new_W + border_px)
#             grid_image[r_start:r_start+resized_img.shape[0], c_start:c_start+resized_img.shape[1]] = resized_img

#         grid_image = cv2.resize(grid_image, (int(2048//1.5), int(1536//1.5)))
#         cv2.imshow("Grid", grid_image)
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             print("[Server] Quitting...")
#             for socket in socket_dict.values():
#                 socket.send_string("quit")
#             break
#         elif key == ord('c'):
#             print("[Server] Sending capture command.")
#             send_capture = True
#             for pc in pc_info.keys():
#                 if capture_state[pc]:
#                     send_capture = False
#                     break
#             if send_capture:
#                 global capture_idx, filename
#                 os.makedirs(os.path.join(shared_dir, "extrinsic", filename, str(capture_idx)), exist_ok=True)
#                 for pc, socket in socket_dict.items():
#                     socket.send_string(f"capture:{capture_idx}")
#                     capture_state[pc] = True
#                 capture_idx += 1

def wait_for_camera_ready():
    while True:
        all_ready = True
        for pc_name, ready in start_dict.items():
            if not ready:
                all_ready = False
                break
        if all_ready:
            break
        time.sleep(0.1)

# Git pull and client run
pc_list = list(pc_info.keys())
git_pull("merging", pc_list)
run_script("python src/demo_250618/client_mingi.py", pc_list)


root_path = "/home/temp_id/shared_data/demo_250618/pringles"
obj_name = root_path.split("/")[-2]

yolo_module = YOLO_MODULE(categories=obj_name)

for pc_name, info in pc_info.items():
    ip = info["ip"]
    sock = context.socket(zmq.DEALER)
    sock.identity = b"server"
    sock.connect(f"tcp://{ip}:5556")
    socket_dict[pc_name] = sock

for pc_name, info in pc_info.items():
    socket_dict[pc_name].send_string("register")
    if socket_dict[pc_name].recv_string() == "registered":
        print(f"[{pc_name}] Registered.")
    
    socket_dict[pc_name].send_string("filename:" + filename)

# Start per-socket listener
for pc_name, sock in socket_dict.items():
    threading.Thread(target=listen_socket, args=(pc_name, sock), daemon=True).start()
wait_for_camera_ready()
# Main UI loop
print("press button")
main_loop(yolo_module)

for pc_name, sock in socket_dict.items():
    sock.send_string("quit")
    sock.close()