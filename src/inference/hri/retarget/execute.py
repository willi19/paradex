import time
import numpy as np
import chime
import argparse
import json
import os, sys

# from paradex.inference.lookup_table import get_traj
from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.signal_generator.UTGE900 import UTGE900
from paradex.io.camera.timecode_receiver import TimecodeReceiver
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.inference.object_6d import get_current_object_6d, get_image
from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.utils.env import get_pcinfo, get_serial_list
from paradex.pose_utils.optimize_initial_frame import object6d_silhouette
from paradex.pose_utils.retarget_utils import get_keypoint_trajectory, visualize_new_trajectory
from paradex.pose_utils.retarget import position_retarget, qpose_dict_to_traj, wrist6d_traj_to_SE3
from paradex.utils.file_io import eef_calib_path, load_latest_eef

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=0, type=int)
    parser.add_argument("--obj_name", required=True)
    parser.add_argument("--vis", action ='store_true')
    parser.add_argument("--no_rot", action="store_true")
    parser.add_argument("--wrist_up", action="store_true")

    # parser.add_argument("--grasp_type", required=True)

    args = parser.parse_args()
    
    path_planning = [(1, 2), (2, 4), (4, 2), (2, 3), (3, 1), (1, 3), (3, 4), (4, 1), (1, 4), (4, 3), (3, 2), (2, 1)]
    
    arm_name = "xarm"
    hand_name = "allegro"
    
    sensors = {}
    # sensors["arm"] = get_arm(arm_name)
    # sensors["hand"] = get_hand(hand_name)
    sensors["signal_generator"] = UTGE900()
    sensors["timecode_receiver"] = TimecodeReceiver()
    
    pc_info = get_pcinfo()
    pc_list = list(pc_info.keys())
    git_pull("merging", pc_list)
    
    scene_idx = args.index
    scene_path = os.path.join(shared_dir, "capture_", "hri", args.obj_name, str(scene_idx))

    # get_image()
    start_6d = get_current_object_6d(obj_name=args.obj_name, marker=True)
    c2r = load_latest_C2R()

    start_6d = c2r @ start_6d
    print(start_6d) # camera space

    # robot space
    hand_trajectory_dict, obj_trajectory_dict = get_keypoint_trajectory(scene_path, start_6d, args.obj_name, no_rot=args.no_rot, wrist_up=args.wrist_up)
    
    q_pose_dict = position_retarget(hand_trajectory_dict)
    # def convert_numpy(obj):
    #     if isinstance(obj, np.float32) or isinstance(obj, np.float64):
    #         return float(obj)
    #     elif isinstance(obj, np.ndarray):
    #         return obj.tolist()  
    #     raise TypeError(f"Type {type(obj)} is not serializable")
    # with open("/home/temp_id/paradex/qpose.json", "w", encoding="utf-8") as file:
    #     json.dump(q_pose_dict, file, ensure_ascii=False, indent = 4, default=convert_numpy)

    wrist_6d, _ = qpose_dict_to_traj(q_pose_dict)
    wrist_6d = wrist6d_traj_to_SE3(wrist_6d) ## 다시 봐야함 base_link 기준이라 6d가

    if args.vis:
        visualize_new_trajectory(args.obj_name, wrist_6d, hand_trajectory_dict, obj_trajectory_dict, q_pose_dict)
   


    
    # pick_traj = np.load(f"{demo_path}/pick.npy")
    # place_traj = np.load(f"{demo_path}/place.npy")
    # pick_hand_traj = np.load(f"{demo_path}/pick_hand.npy")
    # place_hand_traj = np.load(f"{demo_path}/place_hand.npy")
    
    # place_position_list = json.load(open(f"data/lookup/{args.obj_name}/obj_pose.json"))
    start_pos= np.array([[0, 0, 1, 0.3],
                        [1, 0, 0, -0.35],
                        [0, 1, 0, 0.10], 
                        [0, 0, 0, 1]])
    
    end_pos= np.array([[0, 0, 1, 0.25],
                        [1, 0, 0, 0.0],
                        [0, 1, 0, 0.10], 
                        [0, 0, 0, 1]])
    
    eef = load_latest_eef()
    
    import pdb; pdb.set_trace()
    while True:
        sensors["arm"].home_robot(start_pos.copy())  
        home_start_time = time.time()
        while not sensors["arm"].is_ready():
            time.sleep(0.01)

        chime.info()
        
        # traj, hand_traj = get_traj(pick_traj, pick_6D, place_traj, place_6D, pick_hand_traj, place_hand_traj)
        traj, hand_traj = qpose_dict_to_traj(q_pose_dict)
        



        for i in range(len(traj)):
            traj[i] = traj[i] @ np.linalg.inv(eef)
            
        # start the camera
        run_script(f"python src/capture/camera/video_client.py", pc_list)
        sensors["camera"] = RemoteCameraController("video", serial_list=None, sync=True)

        # Set directory
        save_path = os.path.join("retarget", "test", args.obj_name, args.grasp_type)
        shared_path = os.path.join(shared_dir, save_path)
        os.makedirs(shared_path, exist_ok=True)
        if len(os.listdir(shared_path)) == 0:
            capture_idx = 0
        else:
            capture_idx = int(max(os.listdir(shared_path), key=lambda x:int(x))) + 1

        c2r = load_latest_C2R()
        os.makedirs(os.path.join(shared_path, str(capture_idx)))
        copy_calib_files(f'{shared_path}/{capture_idx}')
        # np.save(f'{shared_path}/{capture_idx}/C2R.npy', c2r)
        # np.save(f'{shared_path}/{capture_idx}/pick_6D.npy', pick_6D)
        # np.save(f'{shared_path}/{capture_idx}/place_6D.npy', place_6D)
        # np.save(f'{shared_path}/{capture_idx}/traj.npy', traj)
        # np.save(f'{shared_path}/{capture_idx}/hand_traj.npy', hand_traj)
        
        # Prepare execution
        sensors["arm"].home_robot(traj[0])
        home_start_time = time.time()
        while not sensors["arm"].is_ready():
            if time.time() - home_start_time > 0.5:
                chime.warning()
                home_start_time = time.time()
            time.sleep(0.01)
        
        chime.success()
        
        # Start capture
        sensors['arm'].start(f"{shared_path}/{capture_idx}/raw/{arm_name}")
        sensors['hand'].start(f"{shared_path}/{capture_idx}/raw/{hand_name}")
        sensors['camera'].start(f"{save_path}/{capture_idx}/videos")
        sensors['timecode_receiver'].start(f"{shared_path}/{capture_idx}/raw/timestamp")
        sensors["signal_generator"].on(1)
        
        state_hist = []
        state_time = []
        for i in range(len(traj)):
            sensors["arm"].set_action(traj[i])
            sensors["hand"].set_target_action(hand_traj[i])
            state_hist.append(i)
            state_time.append(time.time())
            time.sleep(0.03)  # Simulate time taken for each action
        
        
        sensors["arm"].end()
        sensors["hand"].end()
        sensors["camera"].end()
        sensors['timecode_receiver'].end()
        sensors['signal_generator'].off(1)
        
        os.makedirs(f"{shared_path}/{capture_idx}/raw/state", exist_ok=True)
        np.save(f"{shared_path}/{capture_idx}/raw/state/state.npy", state_hist)
        np.save(f"{shared_path}/{capture_idx}/raw/state/time.npy", state_time)
        sensors["camera"].quit()
        time.sleep(3) # Need distributor to stop
    
    sensors["arm"].home_robot(end_pos)
    home_start_time = time.time()
    while not sensors["arm"].is_ready():
        time.sleep(0.01)

    chime.info()
    
    for sensor_name, sensor in sensors.items():
        if sensor_name == "camera":
            continue
        sensor.quit()
        
        
        
        
        


    def add_local_frames(
        self,
        traj: dict,  # {frame_idx: 4x4 array}
        axes_length: float = 0.1,
        axes_radius: float = 0.004,
    ):
        """
        traj: {frame_idx: (4,4) numpy array} 형태
        Playback 슬라이더로 재생 가능.
        """
        # dict -> list of keys (정렬 여부 선택 가능)
        frame_ids = list(traj.keys())   # 입력 순서 유지 (Python3.7+)
        self._ensure_playback(len(frame_ids))

        for local_idx, frame_id in enumerate(frame_ids):
            T = np.asarray(traj[frame_id])
            assert T.shape == (4,4), f"Frame {frame_id} has wrong shape {T.shape}"
            
            node = self.server.scene.add_frame(
                f"/frames/t/{local_idx}",
                wxyz=tf.SO3.from_matrix(T[:3,:3]).wxyz,
                position=T[:3,3],
                axes_length=axes_length,
                axes_radius=axes_radius,
                visible=False
            )
            self.frame_nodes[local_idx] = node

        # 첫 프레임 보이도록
        if 0 in self.frame_nodes:
            self.frame_nodes[0].visible = True

        prev_local = 0
        @self.gui_timestep.on_update
        def _(_):
            nonlocal prev_local
            cur = int(self.gui_timestep.value)
            if prev_local in self.frame_nodes:
                self.frame_nodes[prev_local].visible = False
            if cur in self.frame_nodes:
                self.frame_nodes[cur].visible = True
            prev_local = cur