import trimesh
from pathlib import Path
import open3d as o3d
import json
import numpy as np
import sys
import os
import viser


from paradex.pose_utils.io import get_obj_info
from paradex.utils.file_io import load_current_camparam, home_path, shared_dir, load_latest_C2R
from paradex.pose_utils.viser_visualizer import visualize_keypoint_object
from paradex.pose_utils.io import get_camera_params

MEDIA_EDGES = [
    (0,1),(1,2),(2,3),(3,4),           # thumb
    (0,5),(5,6),(6,7),(7,8),           # index
    (5,9),(9,10),(10,11),(11,12),      # middle
    (9,13),(13,14),(14,15),(15,16),    # ring
    (13,17),(17,18),(18,19),(19,20),   # little
    (0,17),                            # wrist to little base
]
def _norm(v):
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n

def _get_R_t(extr):  # extr: (3x4) 또는 (4x4)
    if extr.shape == (4,4):
        R, t = extr[:3,:3], extr[:3,3]
    else:
        R, t = extr[:,:3], extr[:,3]
    return R, t

def compute_world_up_and_front(cam_params, plane_ids, front_cam_id, convention="opencv"):
    """
    return: up_world (3,), front_ref_world (3,)
    - convention="opencv": 카메라 z가 전방(+Z), y는 아래. 카메라 'up' = -y_cam.
    """
    centers = []
    ups_world = []

    for cid in plane_ids:
        if cid not in cam_params: 
            continue
        R, t = _get_R_t(np.asarray(cam_params[cid]["extrinsic"]))
        C = -R.T @ t                      # 카메라 센터(월드)
        centers.append(C)

        if convention == "opencv":
            up_cam = np.array([0., -1., 0.])   # 이미지 위쪽 = -y_cam
        else:  # opengl (보통 -Z가 forward, +Y가 up)
            up_cam = np.array([0.,  1., 0.])
        ups_world.append(R.T @ up_cam)         # 월드에서 본 카메라 up

    centers = np.asarray(centers)
    if centers.shape[0] < 3:
        raise ValueError("Need >=3 cameras to fit a plane.")

    centroid = centers.mean(axis=0)
    U, S, Vt = np.linalg.svd(centers - centroid)
    up_world = _norm(Vt[-1])                   # 평면 법선

    # 부호 정정: 평균 카메라 up과 같은 반구로 향하게
    mean_up = _norm(np.mean(np.asarray(ups_world), axis=0))
    if np.dot(up_world, mean_up) < 0:
        up_world = -up_world

    # front ref: 지정 카메라의 전방
    Rf, tf = _get_R_t(np.asarray(cam_params[front_cam_id]["extrinsic"]))
    if convention == "opencv":
        z_cam_fwd = np.array([0., 0., 1.])     # OpenCV: +Z forward
    else:
        z_cam_fwd = np.array([0., 0., -1.])    # OpenGL: -Z forward

    front_world = Rf.T @ z_cam_fwd
    # up에 수직인 평면으로 투영해서 pitch/roll 성분 제거
    front_ref_world = _norm(front_world - np.dot(front_world, up_world) * up_world)

    return up_world, front_ref_world

def rot_about_axis_4x4(axis, theta):
    a = _norm(axis.astype(np.float64))
    x,y,z = a
    K = np.array([[0,-z,y],[z,0,-x],[-y,x,0]], dtype=np.float64)
    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
    T = np.eye(4); T[:3,:3] = R
    return T

def remove_world_yaw(T_r, up_world, f_local, front_ref_world):
    """
    T_r 객체의 월드 yaw 회전만 제거하고, 위치(translation)는 그대로 보존합니다.
    """
    # 1. 원본 포즈에서 회전(R)과 위치(t)를 분리합니다.
    R_old, t_old = _get_R_t(T_r)

    # 2. 현재 객체의 Yaw 각도를 계산하는 로직 (기존과 동일)
    f_world = R_old @ f_local
    f_proj   = _norm(f_world - np.dot(f_world, up_world)*up_world)
    f_ref    = _norm(front_ref_world - np.dot(front_ref_world, up_world)*up_world)
    
    sinv = np.dot(up_world, np.cross(f_proj, f_ref))
    cosv = float(np.clip(np.dot(f_proj, f_ref), -1.0, 1.0))
    yaw  = np.arctan2(sinv, cosv)

    # 3. Yaw 보정을 위한 순수 3x3 회전 행렬을 만듭니다.
    R_fix_3x3 = rot_about_axis_4x4(up_world, -yaw)[:3, :3]

    # 4. 원본 회전에 보정 회전을 적용하여 새로운 회전을 계산합니다.
    R_new = R_fix_3x3 @ R_old

    # 5. 새로운 회전(R_new)과 '원본' 위치(t_old)를 합쳐 최종 포즈를 만듭니다.
    T_new = np.eye(4)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = t_old
    
    return T_new

def align_rotation_to_vector(T_old, local_front_vector, world_up):
    """
    객체의 '앞(front)' 방향은 유지하되, 그 축을 기준으로 한 회전(roll)을 제거하여
    자세를 일관되게 정렬합니다.

    Args:
        T_old (np.ndarray): 정렬할 원본 객체 포즈 (4x4).
        local_front_vector (np.ndarray): 객체의 로컬 좌표계에서 '앞'으로
                                         간주할 단위 벡터 (기본값: +X축).

    Returns:
        np.ndarray: 회전이 정렬된 새로운 객체 포즈 (4x4).
    """
    # 1. 원본 포즈에서 회전(R)과 이동(t) 정보 분리
    R_old = T_old[:3, :3]
    t_old = T_old[:3, 3]

    # 2. 객체의 '앞' 방향이 현재 월드에서 어느 방향을 가리키는지 계산
    # 이 방향이 새로운 좌표계의 기준 축(예: Z축)이 됩니다.
    front_world = R_old @ local_front_vector
    new_z = front_world / np.linalg.norm(front_world)

    # 3. '빙글빙글 도는 회전'을 없애기 위한 고정된 기준 벡터 설정
    # 월드 좌표계의 '위쪽'([0, 1, 0])을 기준으로 사용합니다.
    world_up = world_up
    
    # 엣지 케이스 처리: 객체가 정확히 위나 아래를 보고 있을 경우,
    # 기준 벡터를 다른 것(예: 월드 X축)으로 바꿉니다.
    if np.abs(np.dot(new_z, world_up)) > 0.999:
        world_up = np.array([1., 0., 0.])

    # 4. 직교 좌표계의 나머지 두 축을 생성
    # new_x: '앞' 방향과 '위' 방향에 모두 수직인 '오른쪽' 방향
    new_x = np.cross(world_up, new_z)
    new_x /= np.linalg.norm(new_x)
    
    # new_y: z축과 x축에 모두 수직인 새로운 '위쪽' 방향
    new_y = np.cross(new_z, new_x)
    
    # 5. 계산된 세 축으로 새로운 회전 행렬(R_new)을 조립
    # 여기서는 Z축을 '앞'으로 하는 일반적인 카메라 좌표계 관례를 따릅니다.
    # 만약 X축을 '앞'으로 하고 싶다면 `np.stack([new_z, new_x, new_y], axis=1)` 등으로 순서를 바꿀 수 있습니다.
    R_new = np.stack([new_x, new_y, new_z], axis=1)

    # 6. 새로운 회전 행렬(R_new)과 기존 이동 벡터(t_old)를 합쳐 최종 포즈 생성
    T_aligned = np.eye(4)
    T_aligned[:3, :3] = R_new
    T_aligned[:3, 3] = t_old
    
    return T_aligned

def fix_pose_flip(T_old, obj_mesh, local_axis_vector, plane_up_vector):
    """
    객체의 실제 기하학적 중심(centroid)을 기준으로 뒤집힘을 보정합니다.
    """
    R_old = T_old[:3, :3]
    t_old = T_old[:3, 3]
    
    world_axis = R_old @ local_axis_vector
    
    if np.dot(world_axis, plane_up_vector) > 0:
        # --- [수정된 부분 시작] ---
        
        # 1. 객체의 로컬 좌표계 기준 centroid를 가져옵니다.
        center_local = obj_mesh.centroid
        
        # 2. 로컬 centroid를 월드 좌표계로 변환하여 회전의 중심점으로 사용합니다.
        center_world = (R_old @ center_local) + t_old
        
        # --- [수정된 부분 끝] ---
        
        rotation_axis = np.cross(world_axis, plane_up_vector)
        if np.linalg.norm(rotation_axis) < 1e-6:
            rotation_axis = np.cross(world_axis, np.array([1., 0., 0.]))
            if np.linalg.norm(rotation_axis) < 1e-6:
                rotation_axis = np.cross(world_axis, np.array([0., 1., 0.]))
        rotation_axis /= np.linalg.norm(rotation_axis)
        
        theta_rad = np.pi
        
        rotation_M = trimesh.transformations.rotation_matrix(
            angle=theta_rad,
            direction=rotation_axis,
            point=center_world  # <-- 수정된 회전 중심점 사용
        )
        
        return rotation_M @ T_old, True
    else:
        return T_old, False

def load_keypoints_from_mano_params(scene_path) -> np.ndarray:
    params_path = os.path.join(scene_path, "mano_params.json")
    with open(params_path, "r") as f:
        mano_params = json.load(f)

    sorted_frames = sorted(mano_params.keys(), key=lambda x: int(x))
    keypoint_3d = np.array(
        [mano_params[frame]["keypoint"][0] for frame in sorted_frames],
        dtype=np.float32,
    )  # (T, 21, 3)
    return keypoint_3d

def get_keypoint_trajectory(scene_path, start_6d, obj_name, no_rot=False):
    obj_initial_T, obj_mesh = None, None
    obj_trajectory = {}
    if obj_name is not None:
        obj_mesh, obj_initial_T, obj_trajectory = get_obj_info(scene_path, obj_name, \
                                                    obj_status_path=None)
    total_frames = len(obj_trajectory.keys()) - 1
    f_local = obj_mesh.principal_inertia_vectors[0, :3].copy()
    f_local /= np.linalg.norm(f_local)
    
    keypoints_raw = load_keypoints_from_mano_params(scene_path)
    old_cam_params, _, _ = get_camera_params(scene_path)

    if no_rot:
        up_world, front_ref_world = compute_world_up_and_front(old_cam_params, 
                                    ["22684737","23022627","22645029","23173281","22641023","22641005"]
                                    , "22645029")
        old_T_inv = np.linalg.inv(obj_trajectory[0]['T'])
        for frame in range(1, total_frames):
            obj_trajectory[frame]['T'] = obj_trajectory[frame]['T'] @ old_T_inv
            
        obj_trajectory[0]['T'], if_flip = fix_pose_flip(obj_trajectory[0]['T'], obj_mesh, f_local, up_world)
        if if_flip:
            obj_trajectory[0]['T'] = remove_world_yaw(obj_trajectory[0]['T'], up_world, np.array([1,0,0]), front_ref_world)
        else:
            obj_trajectory[0]['T'] = remove_world_yaw(obj_trajectory[0]['T'], up_world, np.array([1,0,0]), front_ref_world)
        
        for frame in range(1, total_frames):
            obj_trajectory[frame]['T'] = obj_trajectory[frame]['T'] @ obj_trajectory[0]['T']
    
    out = np.empty((total_frames, 21, 3), dtype=np.float32)
    for frame in range(total_frames):
        keypoint_frame = None
        try:
            keypoint_frame = keypoints_raw[frame]
        except IndexError:
            keypoint_frame = keypoints_raw[frame-1]
        
        ones = np.ones((keypoint_frame.shape[0], 1))
        keypoint_frame = np.hstack([keypoint_frame, ones]) 
        T_obj_inv = np.linalg.inv(obj_trajectory[frame]['T'])
        keypoint_object_coord = (T_obj_inv @ keypoint_frame.T).T[:, :3]
        out[frame] = keypoint_object_coord

    intrinsic, extrinsic = load_current_camparam()
    cam_params = {}
    for cam_id in extrinsic:
        extrinsic_np = np.array(extrinsic[cam_id]) 
        intrinsic_np = np.array(intrinsic[cam_id]['intrinsics_undistort'])
        cam_params[cam_id] = {'extrinsic': extrinsic_np, 'intrinsic':intrinsic_np}

    
    T0_inv = np.linalg.inv(obj_trajectory[0]['T'])
    c2r = load_latest_C2R()
    if no_rot:
        up_world, front_ref_world = compute_world_up_and_front(cam_params, 
                                   ["22684737","23022627","22645029","23173281","22641023","22641005"]
                                   , "22645029")
        start_6d, if_flip = fix_pose_flip(start_6d, obj_mesh, f_local, up_world)
        if if_flip:
            start_6d = remove_world_yaw(start_6d, up_world, np.array([1,0,0]), front_ref_world) #remove_world_yaw(start_6d, up_world, f_local, front_ref_world)
        else:
            start_6d = remove_world_yaw(start_6d, up_world, np.array([1,0,0]), front_ref_world) #remove_world_yaw(start_6d, up_world, f_local, front_ref_world)

    import pdb; pdb.set_trace()
    new_obj_trajectory = {}
    keypoint_dict = {}
    for frame in range(total_frames):
        T_i = obj_trajectory[frame]['T'] @ T0_inv
        T_i = c2r @ T_i @ start_6d
        new_obj_trajectory[frame] = T_i
        
        keypoint_frame = out[frame]
        ones = np.ones((keypoint_frame.shape[0], 1))
        keypoint_frame = np.hstack([keypoint_frame, ones]) 
        keypoint_new_coord = (T_i @ keypoint_frame.T).T[:, :3]
        keypoint_dict[frame] = keypoint_new_coord

    return keypoint_dict, new_obj_trajectory


def visualize_new_trajectory(obj_name, hand_keypoint_dict, obj_trajectory_dict):
    intrinsic, extrinsic = load_current_camparam()
    c2r = load_latest_C2R()
    r2c = np.linalg.inv(c2r)
    cam_params = {}
    for cam_id in extrinsic:
        extrinsic_np = np.array(extrinsic[cam_id]) @ r2c
        intrinsic_np = np.array(intrinsic[cam_id]['intrinsics_undistort'])
        cam_params[cam_id] = {'extrinsic': extrinsic_np, 'intrinsic':intrinsic_np}
    visualize_keypoint_object(obj_name, cam_params, hand_keypoint_dict, obj_trajectory_dict)