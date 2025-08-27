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

def transform_vector(T, v):
    """벡터 (3,) -> (3,)"""
    R = T[:3, :3]                 # rotation part only
    return R @ v

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
    
def align_object_front(T, up_world, front_ref_world, db_local):
    """
    object의 front를 world front 방향에 맞추되,
    up_world 축을 보존하는 회전으로 정렬.

    Args:
        T (np.ndarray): (4,4) object pose (world 좌표계).
        up_world (np.ndarray): (3,) world up vector.
        front_ref_world (np.ndarray): (3,) target world front vector.
        db_local (np.ndarray): (3,) object local front vector.
    
    Returns:
        T_new (np.ndarray): (4,4) 정렬된 object pose.
    """

    # Normalize 입력 벡터들
    up_world = up_world / np.linalg.norm(up_world)
    front_ref_world = front_ref_world / np.linalg.norm(front_ref_world)

    # 현재 object rotation
    R_obj = T[:3,:3]

    # object front in world
    f_world = db_local # db_local
    f_world /= np.linalg.norm(f_world)

    # 회전 각도 계산
    cos_theta = np.dot(f_world, front_ref_world)
    sin_theta = np.dot(np.cross(f_world, front_ref_world), up_world)
    theta = np.arctan2(sin_theta, cos_theta)

    # Rodrigues 공식
    axis = up_world
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R_align = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)

    # 새로운 rotation
    R_new = R_obj @ R_align

    # 새로운 pose
    T_new = T.copy()
    T_new[:3,:3] = R_new
    return T_new

def basis_from_up_front(up, front):
    # up을 단위화
    u = _norm(up)
    # front를 up에 직교한 평면에 투영해 pitch/roll 제거
    f = front - np.dot(front, u) * u
    f = _norm(f)
    # right = up × front  (오른손 좌표계)
    r = _norm(np.cross(u, f))
    # 수치안정 위해 front를 다시 right×up으로 재계산
    f = np.cross(r, u)
    # 열벡터가 축인 3x3 회전행렬 [r u f]
    R = np.stack([r, u, f], axis=1)
    return R
def _project_front_onto_up_plane(front, up):
    # front를 up에 직교한 평면으로 투영 후 정규화
    f = np.asarray(front, float) - np.dot(front, up) * up
    if np.linalg.norm(f) < 1e-9:
        # up과 거의 평행하면 임의의 축으로 보정
        tmp = np.array([1., 0., 0.]) if abs(up[0]) < 0.9 else np.array([0., 1., 0.])
        f = tmp - np.dot(tmp, up) * up
    return _norm(f)

def make_W_from_up_front(up_prev, front_prev, up_today, front_today,
                         anchor_prev=None, anchor_today=None):
    u_p = _norm(up_prev)
    u_t = _norm(up_today)
    if np.dot(u_p, u_t) < 0:
        u_t = -u_t  # up 반구 정렬

    # front를 각 up 평면에 투영 (yaw만 사용)
    f_p = _project_front_onto_up_plane(front_prev,  u_p)
    f_t = _project_front_onto_up_plane(front_today, u_t)

    # ✅ 우손계: right = up × front
    r_p = _norm(np.cross(u_p, f_p))
    r_t = _norm(np.cross(u_t, f_t))

    # ✅ 재직교: front = right × up  (r × u)
    f_p = _norm(np.cross(r_p, u_p))
    f_t = _norm(np.cross(r_t, u_t))

    R_prev  = np.stack([r_p, u_p, f_p], axis=1)   # [right, up, front]
    R_today = np.stack([r_t, u_t, f_t], axis=1)
    R_W = R_today @ R_prev.T

    # (선택) 수치 안전장치: det<0이면 right 뒤집어 우손계 보장
    if np.linalg.det(R_W) < 0:
        r_t = -r_t
        f_t = _norm(np.cross(r_t, u_t))
        R_today = np.stack([r_t, u_t, f_t], axis=1)
        R_W = R_today @ R_prev.T

    # 번역(옵션)
    t_W = np.zeros(3)
    if anchor_prev is not None and anchor_today is not None:
        anchor_prev  = np.asarray(anchor_prev,  float)
        anchor_today = np.asarray(anchor_today, float)
        t_W = anchor_today - R_W @ anchor_prev

    W = np.eye(4)
    W[:3, :3] = R_W
    W[:3,  3] = t_W
    return W


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
    up_local = obj_mesh.principal_inertia_vectors[0, :3].copy()
    up_local /= np.linalg.norm(up_local)
    f_local = obj_mesh.principal_inertia_vectors[1, :3].copy()
    f_local /= np.linalg.norm(f_local)

    keypoints_raw = load_keypoints_from_mano_params(scene_path)
    old_cam_params, _, _ = get_camera_params(scene_path)
    prev_up_world, prev_front_ref_world = compute_world_up_and_front(old_cam_params, 
                                ["22684737","23022627","22645029","23173281","22641023","22641005"]
                                , "22645029")
    if no_rot:

        # object_frame_list.append(up_world)
        # object_frame_list.append(front_ref_world)
        old_T_inv = np.linalg.inv(obj_trajectory[0]['T'])
        for frame in range(1, total_frames):
            obj_trajectory[frame]['T'] = obj_trajectory[frame]['T'] @ old_T_inv
            
        obj_trajectory[0]['T'], if_flip = fix_pose_flip(obj_trajectory[0]['T'], obj_mesh, up_local, prev_up_world)
        db_local = transform_vector(obj_trajectory[0]['T'], f_local)
        db_local = prev_up_world - np.dot(prev_up_world, db_local) * db_local
        db_local /= np.linalg.norm(db_local)
        
        obj_trajectory[0]['T'] = align_object_front(obj_trajectory[0]['T'], prev_up_world, prev_front_ref_world, db_local)
        
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
    
    up_world, front_ref_world = compute_world_up_and_front(cam_params, 
                            ["22684737","23022627","22645029","23173281","22641023","22641005"]
                            , "22645029")
    if no_rot:

        start_6d, if_flip = fix_pose_flip(start_6d, obj_mesh, up_local, up_world)
        db_local = transform_vector(start_6d, f_local)
        db_local = up_world - np.dot(up_world, db_local) * db_local
        db_local /= np.linalg.norm(db_local)
        
        start_6d = align_object_front(start_6d, up_world, front_ref_world, db_local)
    
    W = make_W_from_up_front(prev_up_world, prev_front_ref_world, up_world, front_ref_world)
    import pdb; pdb.set_trace()
    new_obj_trajectory = {}
    keypoint_dict = {}
    for frame in range(total_frames):
        T_i = obj_trajectory[frame]['T'] @ T0_inv
        T_i = W @ T_i @ np.linalg.inv(W)
        T_i = c2r @ T_i @ start_6d
        new_obj_trajectory[frame] = T_i
        
        keypoint_frame = out[frame]
        ones = np.ones((keypoint_frame.shape[0], 1))
        keypoint_frame = np.hstack([keypoint_frame, ones]) 
        keypoint_new_coord = (T_i @ keypoint_frame.T).T[:, :3]
        keypoint_dict[frame] = keypoint_new_coord
    

    import pdb; pdb.set_trace()
    return keypoint_dict, new_obj_trajectory


def visualize_new_trajectory(obj_name, hand_keypoint_dict, obj_trajectory_dict, q_pose_dict):
    intrinsic, extrinsic = load_current_camparam()
    c2r = load_latest_C2R()
    r2c = np.linalg.inv(c2r)
    cam_params = {}
    for cam_id in extrinsic:
        extrinsic_np = np.array(extrinsic[cam_id]) @ r2c
        intrinsic_np = np.array(intrinsic[cam_id]['intrinsics_undistort'])
        cam_params[cam_id] = {'extrinsic': extrinsic_np, 'intrinsic':intrinsic_np}
    visualize_keypoint_object(obj_name, cam_params, hand_keypoint_dict, obj_trajectory_dict, q_pose_dict)