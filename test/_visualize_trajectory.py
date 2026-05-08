import os
import pickle
import numpy as np
import trimesh

import torch
import sys, os
from pathlib import Path

from paradex.visualization.visualizer.viser import ViserViewer


# IsaacLab에서 생성한 trajectory pickle 경로
COMPUTED_TRAJ_PATH = "/home/robot/shared_data/jisoo_test/computed_trajectory.pickle"

# xArm + Allegro 로봇 URDF (기존 스크립트와 동일 경로 사용)
ROBOT_URDF = "/home/robot/shared_data/jisoo_test/robot_configuration/xarm_allegro/xarm_allegro.urdf"

curobo_joint_names =  ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'thumb_base', 'thumb_proximal', 'thumb_medial', 'thumb_distal', 'index_base', 'index_proximal', 'index_medial', 'index_distal', 'middle_base', 'middle_proximal', 'middle_medial', 'middle_distal', 'ring_base', 'ring_proximal', 'ring_medial', 'ring_distal']
isaac_joint_names =  ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'index_base', 'middle_base', 'ring_base', 'thumb_base', 'index_proximal', 'middle_proximal', 'ring_proximal', 'thumb_proximal', 'index_medial', 'middle_medial', 'ring_medial', 'thumb_medial', 'index_distal', 'middle_distal', 'ring_distal', 'thumb_distal']


def rearrange_joint_pose(cur_pose, cur_joint_names, target_joint_names):
    '''
        cur_pose: DxN
    '''
    if torch.is_tensor(cur_pose):
        new_pose = torch.zeros((cur_pose.shape[0], len(target_joint_names)), device=cur_pose.device)
    else:
        new_pose = np.zeros((cur_pose.shape[0], len(target_joint_names)))
    for dix, dname in enumerate(target_joint_names):
        if dname in cur_joint_names:
            org_idx = cur_joint_names.index(dname)
            new_pose[:,dix] = cur_pose[:,org_idx]
    return new_pose

def load_computed_trajectory(path: str = COMPUTED_TRAJ_PATH, idx=0):
    """IsaacLab에서 계산한 trajectory pickle을 로드해서 (T, dof) numpy 배열로 반환."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, (list, tuple)) or len(data) == 0:
        raise ValueError(f"computed_trajectory at {path} is empty or invalid.")

    # 유효한 traj가 들어있는 항목만 필터링
    valid_items = [
        item
        for item in data
        if isinstance(item, dict) and "traj" in item and item["traj"] is not None
    ]
    if len(valid_items) == 0:
        raise ValueError("No valid 'traj' entries found in computed_trajectory.")

    if idx >= len(valid_items):
        print(f"[INFO] Index {idx} is out of range, using the last index {len(valid_items)-1}")
        idx = len(valid_items)-1

    valid_data = valid_items[idx]
    traj = valid_data["traj"]

    # torch.Tensor / list 등을 numpy 배열로 통일
    try:
        import torch

        if isinstance(traj, torch.Tensor):
            traj = traj.detach().cpu().numpy()
    except ImportError:
        pass

    if isinstance(traj, list):
        traj = np.array(traj)

    # (B, T, dof) 형태면 첫 번째 배치만 사용
    if traj.ndim == 3:
        traj = traj[0]

    if traj.ndim != 2:
        raise ValueError(f"Unexpected traj shape: {traj.shape}, expected (T, dof).")

    return traj, data


def main(idx=0):
    # trajectory 로드
    traj, data = load_computed_trajectory(COMPUTED_TRAJ_PATH, idx)
    print(f"[INFO] Loaded trajectory from {COMPUTED_TRAJ_PATH}, shape={traj.shape}")

    if idx >= len(data):
        print(f"[INFO] Index {idx} is out of range, using the last index {len(data)-1}")
        idx = len(data)-1
        
    valid_data = data[idx]
    print(
        "[INFO] First element:",
        "hand_pose_idx =", valid_data.get("hand_pose_idx"),
        "ee_pos =", valid_data.get("ee_pos"),
    )

    vis = ViserViewer()

    # 필요하다면 object mesh와 pose를 여기서 추가할 수 있음
    # (현재 computed_trajectory에는 object 정보가 없으므로 로봇만 시각화)
    # 예시:
    # obj_mesh_path = "/path/to/object.obj"
    # mesh = trimesh.load(obj_mesh_path)
    # obj_pose = np.eye(4)  # 4x4 SE3
    # vis.add_object("object", mesh, obj_pose)

    # 로봇과 trajectory 추가
    vis.add_robot("robot", ROBOT_URDF)
    vis.add_traj("traj", {"robot": rearrange_joint_pose(traj, isaac_joint_names, curobo_joint_names)})
    vis.add_floor(height=0.0)
    vis.start_viewer()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()  
    main(args.idx)