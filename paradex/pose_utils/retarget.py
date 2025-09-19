from paradex.pose_utils.dex_retargeting.dex_retargeting.constants import (
    RobotName, HandType, RetargetingType, get_default_config_path
)
from paradex.pose_utils.dex_retargeting.dex_retargeting.retargeting_config import RetargetingConfig

import numpy as np

def build_allegro_retargeter_right():
    cfg_path = get_default_config_path(
        RobotName.allegro, RetargetingType.position, HandType.right
    )
    print(cfg_path)
    return RetargetingConfig.load_from_file(cfg_path).build()

def position_retarget(hand_keypoint_dict):
    retargeter = build_allegro_retargeter_right()
    indices = retargeter.optimizer.target_link_human_indices
    num_frames = len(hand_keypoint_dict.keys())
    
    for i in range(10):
        retargeter.retarget(hand_keypoint_dict[0][indices, :])
    
    q_dict = {}
    for frame in range(0, num_frames):
        q = retargeter.retarget(hand_keypoint_dict[frame][indices, :])
        q_dict[frame] = q
        
    return q_dict

def qpose_dict_to_traj(q_pose_dict):
    """
    q_pose_dict: {frame_idx: (22,) or similar ndarray}
    
    return:
        q_pose_dict (그대로),
        q_hand_dict: {frame_idx: (16,) reordered}
    """
    q_hand_dict = {}
    q_6d_dict = {}

    for key, qv in q_pose_dict.items():
        qv = np.asarray(qv)
        q_hand = qv[6:]  # shape (16,)
        q_hand_reordered = np.zeros_like(q_hand)

        # 매핑
        q_hand_reordered[0:4]   = q_hand[0:4]
        q_hand_reordered[4:12]  = q_hand[8:16]
        q_hand_reordered[12:16] = q_hand[4:8]

        q_hand_dict[key] = q_hand_reordered
        q_6d_dict[key] = qv[0:6]

    return q_6d_dict, q_hand_dict
from scipy.spatial.transform import Rotation as R


def wrist6d_to_SE3(q6):
    """
    q6 : (6,) numpy array
         [x, y, z, roll, pitch, yaw]
    return : (4,4) homogeneous transform matrix
    """
    x, y, z, roll, pitch, yaw = q6

    # 회전행렬
    rot = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

    # 4x4 변환행렬
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [x, y, z]

    return T

def wrist6d_traj_to_SE3(q_dict: dict) -> dict:
    """
    q_dict : {frame_idx: (6,) numpy array} trajectory of wrist 6DOF (현재 base_link 기준)
    return : {frame_idx: (4,4) numpy array} trajectory of SE(3) (wrist 기준)
    """
    # base_link -> wrist 고정 변환 (URDF에서 palm_joint + wrist_joint의 xyz 합산)
    T_base_wrist = np.eye(4)
    T_base_wrist[:3, 3] = [0.0, 0.0, 0.0]   # -0.065 -0.03

    T_dict = {}
    for k, q in q_dict.items():
        q = np.asarray(q).reshape(6,)
        T_world_base = wrist6d_to_SE3(q)  # (4,4), world->base_link
        T_world_wrist = T_world_base @ T_base_wrist
        T_dict[k] = T_world_wrist
    return T_dict



import numpy as np
import open3d as o3d
import time


def play_local_frames(traj: np.ndarray, axis_len: float = 0.05, fps: int = 10):
    """
    traj: (N,4,4) numpy array
    axis_len: 좌표축 길이
    fps: 초당 프레임 재생 속도
    """
    assert traj.ndim == 3 and traj.shape[1:] == (4,4), "traj shape must be (N,4,4)"

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Local Frame Player")

    # 기준 world 좌표축
    world_cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_len*2.0)
    vis.add_geometry(world_cf)

    # 현재 프레임 좌표축 (업데이트용)
    current_cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_len)
    vis.add_geometry(current_cf)

    dt = 1.0 / fps
    for i in range(traj.shape[0]):
        # 좌표축을 새로운 것으로 교체 (transform 누적 방지)
        vis.remove_geometry(current_cf, reset_bounding_box=False)
        current_cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_len)
        current_cf.transform(traj[i])
        vis.add_geometry(current_cf)

        vis.poll_events()
        vis.update_renderer()
        time.sleep(dt)

    vis.destroy_window()