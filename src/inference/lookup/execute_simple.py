import time
import numpy as np
from threading import Event
import argparse
import json
import os
from datetime import datetime
import cv2
import trimesh
import copy

from paradex.inference.simulate import simulate
from paradex.inference.lookup_table import get_traj
from paradex.inference.util import home_robot
from paradex.inference.object_6d import get_current_object_6d, normalize_cylinder, get_goal_position

from paradex.io.camera.util import get_image
from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.signal_generator.UTGE900 import UTGE900
from paradex.io.camera.timecode_receiver import TimecodeReceiver
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.connect import run_script

from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R, load_current_camparam, rsc_path
from paradex.utils.env import get_pcinfo
from paradex.utils.keyboard_listener import listen_keyboard

from paradex.visualization_.renderer import BatchRenderer
from paradex.image.projection import get_cammtx, project_point, project_mesh, project_mesh_nvdiff
from paradex.image.overlay import overlay_mask
from paradex.image.merge import merge_image

use_sim = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", required=True)
    parser.add_argument("--hand", required=True)
    parser.add_argument("--marker", default=False, action="store_true")

    args = parser.parse_args()
    
    pc_info = get_pcinfo()
    pc_list = list(pc_info.keys())

    arm_name = "xarm"
    hand_name = args.hand
    
    sensors = {}
    sensors["arm"] = get_arm(arm_name)
    sensors["hand"] = get_hand(hand_name)
    
    run_script(f"python src/capture/camera/image_client.py", pc_list)
    sensors["camera"] = RemoteCameraController("image", None)
    
    start_pos= np.array([[0, 0, 1, 0.3],
                        [1, 0, 0, -0.4],
                        [0, 1, 0, 0.10], 
                        [0, 0, 0, 1]])
    
    end_pos= np.array([[0, 0, 1, 0.25],
                        [1, 0, 0, 0.0],
                        [0, 1, 0, 0.10], 
                        [0, 0, 0, 1]])
    
    c2r = load_latest_C2R()
    intrinsic, extrinsic = load_current_camparam()
    extrinsic_list = []
    intrinsic_list = []
    
    serial_list = list(extrinsic.keys())
    serial_list.sort()
    for serial_name in serial_list:
        extmat = extrinsic[serial_name]
        extrinsic_list.append(extmat @ c2r)
        
        intrinsic_list.append(intrinsic[serial_name]['intrinsics_undistort'])
    
    renderer = BatchRenderer(intrinsic_list, extrinsic_list, width=2048, height=1536, device='cuda')
    mesh = trimesh.load(os.path.join(rsc_path, "object", args.object, f"{args.object}.obj"))
    save_path = os.path.join("inference", "lookup", args.object)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    place_id_list = ["1"]
    
    sensors["camera"].start(f"shared_data/inference/register_{timestamp}")
    sensors["camera"].end()
    
    register_dict = {}
    for img_name in os.listdir(f"{shared_dir}/inference/register_{timestamp}"):
        serial_num = img_name.split(".")[0]
        register_dict[serial_num] = cv2.imread(f"{shared_dir}/inference/register_{timestamp}/{img_name}")
    
    place_position_dict = get_goal_position(register_dict, place_id_list)
    
    shared_path = os.path.join(shared_dir, save_path)
    os.makedirs(shared_path, exist_ok=True)
    if len(os.listdir(shared_path)) == 0:
        capture_idx = 0
    else:
        capture_idx = int(max(os.listdir(shared_path), key=lambda x:int(x))) + 1
    
    stop_event = Event()
    start_event = Event()
 
    event_dict = {"q":stop_event, "y":start_event}
    listen_keyboard(event_dict)
    
    while not stop_event.is_set():
        place_id = place_id_list[int(capture_idx) % len(place_id_list)]
        place_6D = place_position_dict[place_id]
        
        home_robot(sensors["arm"], start_pos)

        print("press y after fixing object position")        
        while not start_event.is_set() and not stop_event.is_set():
            time.sleep(0.1)
        
        if stop_event.is_set():
            break
        
        start_event.clear()
        
        sensors["camera"].start(os.path.join("shared_data", save_path, str(capture_idx), "pick"))
        sensors["camera"].end()
        
        pick_img_dir = os.path.join(shared_path, str(capture_idx), "pick")       
        os.makedirs(pick_img_dir, exist_ok=True) 
        img_dict = {img_name.split(".")[0]:cv2.imread(os.path.join(pick_img_dir, img_name)) for img_name in os.listdir(pick_img_dir)}
        
        pick_6D = get_current_object_6d(args.object, args.marker, img_dict)
        pick_6d = normalize_cylinder(pick_6D)

        choosen_index, traj, hand_traj = get_traj(args.object, hand_name, start_pos.copy(), pick_6D.copy(), place_6D.copy())
        
        # Show simulation
        if use_sim:
            print("press y if trajectory ok")
            simulate(traj, hand_traj, pick_6D, place_6D, hand_name, args.object, start_event, stop_event)
            
            if stop_event.is_set():
                break
            
            start_event.clear()
        
        # start the camera
        copy_calib_files(f'{shared_path}/{capture_idx}')
        np.save(f'{shared_path}/{capture_idx}/C2R.npy', c2r)
        
        np.save(f'{shared_path}/{capture_idx}/pick_6D.npy', pick_6D)
        np.save(f'{shared_path}/{capture_idx}/target_6D.npy', place_6D)
        np.save(f'{shared_path}/{capture_idx}/traj.npy', traj)
        np.save(f'{shared_path}/{capture_idx}/hand_traj.npy', hand_traj)
        np.save(f'{shared_path}/{capture_idx}/choosen_index.npy', np.array(choosen_index))
        
        # Start capture
        sensors['arm'].start(f"{shared_path}/{capture_idx}/raw/{arm_name}")
        sensors['hand'].start(f"{shared_path}/{capture_idx}/raw/{hand_name}")
        
        state_hist = []
        state_time = []
        for i in range(len(traj)):
            if i == len(traj) - 100:
                sensors["camera"].start(os.path.join("shared_data", save_path, str(capture_idx), "place"))
                sensors["camera"].end()
                
            sensors["arm"].set_action(traj[i])
            sensors["hand"].set_target_action(hand_traj[i])
            state_hist.append(i)
            state_time.append(time.time())
            time.sleep(0.03)  # Simulate time taken for each action
            
            
            if stop_event.is_set():
                break
        
        sensors["arm"].end()
        sensors["hand"].end()
        
        
        place_img_dir = os.path.join(shared_path, str(capture_idx), "place")        
        img_dict = {img_name.split(".")[0]:cv2.imread(os.path.join(place_img_dir, img_name)) for img_name in os.listdir(place_img_dir)}
                                                                   
        cur_6D = get_current_object_6d(args.object, args.marker, img_dict)
        # print(cur_6D)
        os.makedirs(f"{shared_path}/{capture_idx}/raw/state", exist_ok=True)
        np.save(f"{shared_path}/{capture_idx}/raw/state/state.npy", state_hist)
        np.save(f"{shared_path}/{capture_idx}/raw/state/time.npy", state_time)
        np.save(f'{shared_path}/{capture_idx}/place_6D.npy', cur_6D)
        
        time.sleep(1) # Need distributor to stop
        
        # transformed_mesh = copy.deepcopy(mesh)
        # transformed_mesh.apply_transform(place_6D)
        
        # frame, mask = project_mesh_nvdiff(transformed_mesh, renderer)
        # mask = mask.detach().cpu().numpy()[:,:,:,0].astype(np.bool_)
        
        # os.makedirs(f"{shared_path}/{capture_idx}/overlay", exist_ok=True)
        # for i, serial_num in enumerate(serial_list):
        #     overlay_mask(img_dict[serial_num], mask[i], 0.7, np.array((255,0, 0)))
        #     cv2.imwrite(f"{shared_path}/{capture_idx}/overlay/{serial_num}.png",img_dict[serial_num])
        
        # cv2.imwrite(f"{shared_path}/{capture_idx}/merge.png",merge_image(img_dict))
        
        # top_view = np.ones((800, 1600, 3), dtype=np.uint8) * 255
        
        # cur_pos = (800 - int(cur_6D[0,3]*1000), 1600 - int(cur_6D[1,3]*1000))  # 스케일 조정
        # place_pos = (800 - int(place_6D[0,3]*1000), 1600 - int(place_6D[1,3]*1000))
        
        # success = "success" if cur_6D[2,2] > 0.7 else "fail"
        # cv2.circle(top_view, cur_pos, 35, (0,0,255), 5)
        # cv2.circle(top_view, place_pos, 35, (0,255,0), 5)
        
        # # 거리 계산 (mm 단위)
        # distance_mm = np.linalg.norm(np.array(cur_6D[:2,3]) - np.array(place_6D[:2,3])) * 1000

        # # 두 점 사이에 선 그리기
        # cv2.line(top_view, cur_pos, place_pos, (255, 255, 255), 2)

        # # 텍스트 추가
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 0.8
        # thickness = 2
        
        # # 색상 설명
        # cv2.putText(top_view, "Red: Current Position", (50, 50), font, font_scale, (0, 0, 255), thickness)
        # cv2.putText(top_view, "Green: Target Position", (50, 90), font, font_scale, (0, 255, 0), thickness)

        # # 거리 정보
        # cv2.putText(top_view, f"Distance: {distance_mm:.1f} mm", (50, 130), font, font_scale, (255, 255, 255), thickness)
        # cv2.putText(top_view, "Scale: 0.8m x 1.6m", (50, 750), font, 0.8, (128, 128, 128), 1)
        # cv2.putText(top_view, f"{success}", (50, 700), font, 0.8, (128, 128, 128), 1)
        # cv2.arrowedLine(top_view, (800, 750), (800, 600), (0,0,0), 5, tipLength=0.25)
        
        # cv2.imwrite(f"{shared_path}/{capture_idx}/top_view.png",top_view)
        
        capture_idx += 1
        
    home_robot(sensors["arm"], end_pos.copy())
    for sensor_name, sensor in sensors.items():
        sensor.quit()