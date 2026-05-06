import os
import cv2
import argparse
import numpy as np
import tqdm
from copy import deepcopy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from paradex.utils.file_io import find_latest_directory

from paradex.calibration.utils import handeye_calib_path, load_camparam
from paradex.calibration.Tsai_Lenz import solve_ax_xb, solve_axb_cpu
from paradex.robot.utils import get_robot_urdf_path
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.image.image_dict import ImageDict
from paradex.image.aruco import merge_charuco_detection, find_common_indices, detect_charuco
from paradex.transforms.conversion import SOLVE_XA_B
from paradex.visualization.robot import RobotModule


def refine_handeye_joint(name, X_init):
    """Refine X by jointly fitting X and per-marker eef-frame positions to all
    triangulated observations.

    AX=XB only constrains relative motions; absolute X can carry several mm of
    bias. Minimizing |p_cam_im - X @ eef_i @ p_marker_m|^2 over X (6 DoF) and
    {p_marker_m} typically removes that bias.
    """
    root_dir = os.path.join(handeye_calib_path, name)
    index_list = sorted(os.listdir(root_dir))
    eef_list = [np.load(os.path.join(root_dir, idx, "eef_fk.npy")) for idx in index_list]
    ids_list = [np.load(os.path.join(root_dir, idx, "charuco_3d_ids.npy")) for idx in index_list]
    cor_list = [np.load(os.path.join(root_dir, idx, "charuco_3d_corners.npy")) for idx in index_list]

    all_ids = sorted({int(m) for ids in ids_list for m in ids})
    mid2idx = {m: i for i, m in enumerate(all_ids)}

    obs = []
    for k in range(len(index_list)):
        for mid, cor in zip(ids_list[k], cor_list[k]):
            obs.append((k, mid2idx[int(mid)], cor.astype(np.float64)))

    # Init marker_in_eef from X_init averaged across frames
    mp = {m: [] for m in all_ids}
    for k in range(len(index_list)):
        eef_inv = np.linalg.inv(eef_list[k])
        X_inv = np.linalg.inv(X_init)
        for mid, cor in zip(ids_list[k], cor_list[k]):
            h = np.array([cor[0], cor[1], cor[2], 1.0])
            p_eef = (eef_inv @ X_inv @ h)[:3]
            mp[int(mid)].append(p_eef)
    P_init = np.array([np.mean(mp[m], axis=0) for m in all_ids])

    eef_arr = np.array(eef_list)

    def unpack(x):
        R = Rot.from_rotvec(x[:3]).as_matrix()
        t = x[3:6]
        P = x[6:].reshape(-1, 3)
        return R, t, P

    def residual(x):
        R, t, P = unpack(x)
        X4 = np.eye(4); X4[:3, :3] = R; X4[:3, 3] = t
        out = np.empty(len(obs) * 3)
        for n, (k, m, p_cam) in enumerate(obs):
            p_eef_h = np.array([P[m, 0], P[m, 1], P[m, 2], 1.0])
            out[3*n:3*n+3] = (X4 @ eef_arr[k] @ p_eef_h)[:3] - p_cam
        return out

    x0 = np.concatenate([
        Rot.from_matrix(X_init[:3, :3]).as_rotvec(),
        X_init[:3, 3],
        P_init.flatten(),
    ])
    r0 = residual(x0)
    e0 = np.linalg.norm(r0.reshape(-1, 3), axis=1) * 1000
    print(f"Joint refine init  : 3D median={np.median(e0):.3f}mm mean={np.mean(e0):.3f}mm max={np.max(e0):.3f}mm")

    res = least_squares(residual, x0, method='lm', max_nfev=2000)
    R_f, t_f, _ = unpack(res.x)
    X_f = np.eye(4); X_f[:3, :3] = R_f; X_f[:3, 3] = t_f

    e1 = np.linalg.norm(residual(res.x).reshape(-1, 3), axis=1) * 1000
    print(f"Joint refine final : 3D median={np.median(e1):.3f}mm mean={np.mean(e1):.3f}mm max={np.max(e1):.3f}mm")

    delta_t = np.linalg.norm(t_f - X_init[:3, 3]) * 1000
    R_delta = R_f @ X_init[:3, :3].T
    cos_arg = np.clip((np.trace(R_delta) - 1) / 2, -1.0, 1.0)
    delta_deg = np.degrees(np.arccos(cos_arg))
    print(f"Joint refine moved : translation={delta_t:.2f}mm, rotation={delta_deg:.3f}°")
    return X_f

def undistort_and_detect_charuco(name):
    img_dict = None
    root_dir = os.path.join(handeye_calib_path, name)
    index_list = sorted(os.listdir(root_dir))
    
    for index in tqdm.tqdm(index_list, desc="Undistort and detect charuco"):
        print(f"Processing index {index}...")
        if os.path.exists(os.path.join(root_dir, index, "charuco_3d_ids.npy")) and \
           os.path.exists(os.path.join(root_dir, index, "charuco_3d_corners.npy")):
            continue

        # if os.path.exists(os.path.join(root_dir, index, "undistort", "images")) and \
        #     len(os.listdir(os.path.join(root_dir, index, "undistort", "images"))) == \
        #     len(os.listdir(os.path.join(root_dir, index, "images"))):
        #     continue
        
        os.makedirs(os.path.join(root_dir, index, "undistort", "images"), exist_ok=True)
        if img_dict is None:
            img_dict = ImageDict.from_path(os.path.join(root_dir, index))
        else:
            img_dict.update_path(os.path.join(root_dir, index))
            
        undistort_img_dict = img_dict.undistort(save_path=os.path.join(root_dir, index, "undistort"))
        
        charuco_3d = undistort_img_dict.triangulate_charuco()
        charuco_3d = merge_charuco_detection(charuco_3d)
        
        charuco_2d = undistort_img_dict.apply(detect_charuco, False)
        detection = {}
        for serial in charuco_2d:
            detection[serial] = merge_charuco_detection(charuco_2d[serial])['checkerCorner']
        
        detectionDict = undistort_img_dict.draw_keypoint(detection, color=(0,255,0))
        print(charuco_3d['checkerCorner'].shape)
        if len(charuco_3d['checkerCorner'])<=0:
            continue
        projected_dict = undistort_img_dict.project_pointcloud(charuco_3d['checkerCorner'])
        detectionDict = detectionDict.draw_keypoint(projected_dict, color=(255,0,0))
        detectionDict.save(os.path.join(root_dir, index, "detection"))
        
        np.save(os.path.join(root_dir, index, "charuco_3d_ids.npy"), charuco_3d['checkerIDs'])
        np.save(os.path.join(root_dir, index, "charuco_3d_corners.npy"), charuco_3d['checkerCorner'])

def compute_fk(name, arm):
    root_dir = os.path.join(handeye_calib_path, name)
    index_list = sorted(os.listdir(root_dir))

    robot_wrapper = RobotWrapper(get_robot_urdf_path(arm_name=arm))

    for index in index_list:
        if os.path.exists(os.path.join(root_dir, index, "eef_fk.npy")):
            continue
        
        qpos = np.load(os.path.join(root_dir, index, "qpos.npy"))
        eef = robot_wrapper.compute_forward_kinematics(qpos, link_list=["link6"])['link6']
        np.save(os.path.join(root_dir, index, "eef_fk.npy"), eef)

def compute_motion(name, min_corners=8, min_rotation_deg=5.0, rot_match_tol_deg=2.0):
    """Build relative-motion pairs for hand-eye AX=XB.

    Charuco corners are coplanar, so Kabsch SVD on a small/near-planar set
    can flip the rotation. Filter out:
      - pairs with too few common corners (planar ambiguity)
      - pairs with too little rotation (Tsai-Lenz needs excitation)
      - pairs where cam-frame and robot-frame rotation magnitudes disagree
        beyond `rot_match_tol_deg` (sanity check: marker is rigid on eef,
        so |R_cam| must equal |R_robot|).
    """
    def _angle(R):
        return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))

    motion_wrt_cam = []
    motion_wrt_robot = []

    root_dir = os.path.join(handeye_calib_path, name)
    index_list = os.listdir(root_dir)
    index_list.sort()

    eef_list = [np.load(os.path.join(root_dir, index, "eef_fk.npy")) for index in index_list]
    charuco_id_list = [np.load(os.path.join(root_dir, index, "charuco_3d_ids.npy")) for index in index_list]
    charuco_cor_list = [np.load(os.path.join(root_dir, index, "charuco_3d_corners.npy")) for index in index_list]

    for i in range(1, len(index_list)):
        eef = eef_list[i]
        eef_prev = eef_list[i - 1]
        M_rob = eef_prev @ np.linalg.inv(eef)

        ids = charuco_id_list[i]
        ids_prev = charuco_id_list[i - 1]
        common_idx, common_idx_prev = find_common_indices(ids, ids_prev)

        if len(common_idx) < min_corners:
            print(f"Motion {i-1}->{i} SKIP: only {len(common_idx)} common corners")
            continue

        cam_cor = charuco_cor_list[i][common_idx]
        cam_cor_prev = charuco_cor_list[i - 1][common_idx_prev]
        M_cam = SOLVE_XA_B(cam_cor, cam_cor_prev)

        a_cam = _angle(M_cam[:3, :3])
        a_rob = _angle(M_rob[:3, :3])
        err = cam_cor_prev - (M_cam[:3, :3] @ cam_cor.T).T - M_cam[:3, 3]
        fit_mm = np.mean(np.linalg.norm(err, axis=1)) * 1000

        if a_rob < min_rotation_deg:
            print(f"Motion {i-1}->{i} SKIP: |R_rob|={a_rob:.2f}° below {min_rotation_deg}°")
            continue
        if abs(a_cam - a_rob) > rot_match_tol_deg:
            print(f"Motion {i-1}->{i} SKIP: rotation mismatch |R_cam|={a_cam:.2f}° vs |R_rob|={a_rob:.2f}°")
            continue

        motion_wrt_cam.append(M_cam)
        motion_wrt_robot.append(M_rob)
        print(f"Motion {i-1}->{i} keep: fit={fit_mm:.2f}mm |R|={a_rob:.2f}° n={len(common_idx)}")

    print(f"compute_motion: kept {len(motion_wrt_cam)} / {len(index_list)-1} pairs")
    return motion_wrt_cam, motion_wrt_robot

def debug(name, arm):
    root_dir = os.path.join(handeye_calib_path, name)

    index_list = os.listdir(root_dir)
    index_list.sort()

    robot_wrt_cam = np.load(os.path.join(root_dir, index_list[0], "C2R.npy")) # cam_wrt_robot
    marker_pos = {}
    
    rm = RobotModule(get_robot_urdf_path(arm_name=arm))
    intrinsic, extrinsic = load_camparam(os.path.join(root_dir, "0"))
    
    for index in index_list:
        eef = np.load(os.path.join(root_dir, index, "eef_fk.npy"))
        eef_from_robot = np.load(os.path.join(root_dir, index, "eef.npy"))
        # FK error
        eef_error = np.linalg.inv(eef) @ eef_from_robot
        trans_error = np.linalg.norm(eef_error[:3, 3]) * 1000
        
        # Rotation error (degrees)
        R_error = eef_error[:3, :3]
        angle_error = np.arccos((np.trace(R_error) - 1) / 2) * 180 / np.pi
        
        print(f"fk error {index}: trans={trans_error:.2f}mm, rot={angle_error:.2f}deg")
        
        charuco_3d_cor = np.load(os.path.join(root_dir, index, "charuco_3d_corners.npy"))
        charuco_id_cor = np.load(os.path.join(root_dir, index, "charuco_3d_ids.npy"))
        
        for mid, cor in zip(charuco_id_cor, charuco_3d_cor):
            if mid not in marker_pos:
                marker_pos[mid] = []
            cor_h = np.ones((4,))
            cor_h[:3] = cor
            marker_pos[mid].append(((np.linalg.inv(eef) @ np.linalg.inv(robot_wrt_cam) @ cor_h.T).T)[:3])

    marker_wrt_eef = []
    for mid in marker_pos:
        pos = np.array(marker_pos[mid])
        mean_pos = np.mean(pos, axis=0)
        std_pos = np.std(pos, axis=0)
        print(f"Marker {mid}: std={std_pos}, mean={mean_pos}")

        marker_wrt_eef.append(mean_pos)
    marker_wrt_eef = np.array(marker_wrt_eef)
    
    img_dict = None
    for index in tqdm.tqdm(index_list, desc="Debug"):
        # if os.path.exists(os.path.join(root_dir, index, "debug", 'images')) and \
        #    len(os.listdir(os.path.join(root_dir, index, "debug", 'images'))) == \
        #    len(os.listdir(os.path.join(root_dir, index, "images"))):
        #     continue
        
        if img_dict is None:
            img_dict = ImageDict.from_path(os.path.join(root_dir, index, "undistort"))
            img_dict.set_camparam(intrinsic, extrinsic)
        else:
            img_dict.update_path(os.path.join(root_dir, index, "undistort"))
        
        qpos = np.load(os.path.join(root_dir, index, "qpos.npy"))
        eef = np.load(os.path.join(root_dir, index, "eef_fk.npy"))
        
        rm.update_cfg(qpos)
        robot_mesh = rm.get_robot_mesh()
        robot_mesh.apply_transform(robot_wrt_cam)
        
        overlay_img_dict = img_dict.project_mesh(robot_mesh)
        
        marker_wrt_eef_h = np.ones((marker_wrt_eef.shape[0], 4))
        marker_wrt_eef_h[:,:3] = marker_wrt_eef 

        marker_wrt_cam = (robot_wrt_cam @ eef @ marker_wrt_eef_h.T).T[:, :3]
        proj_marker = overlay_img_dict.project_pointcloud(marker_wrt_cam)
        overlay_img_dict = overlay_img_dict.draw_keypoint(proj_marker, (255, 0, 0))

        marker_3d  = np.load(os.path.join(root_dir, index, "charuco_3d_corners.npy"))
        proj_marker_3d = overlay_img_dict.project_pointcloud(marker_3d)
        overlay_img_dict = overlay_img_dict.draw_keypoint(proj_marker_3d, (0, 0, 255))

        charuco_2d = img_dict.apply(detect_charuco, False)
        detection_2d = {}
        for serial in charuco_2d:
            detection_2d[serial] = merge_charuco_detection(charuco_2d[serial])['checkerCorner']
        overlay_img_dict = overlay_img_dict.draw_keypoint(detection_2d, (0, 255, 0))

        overlay_img_dict.save(os.path.join(root_dir, index, "debug"))
                
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default=None, help="Name of the calibration directory.")
parser.add_argument("--arm", type=str, default="xarm", help="Name of the robot arm.")

args = parser.parse_args()
if args.name is None:
    args.name = find_latest_directory(handeye_calib_path)

name = args.name
root_path = os.path.join(handeye_calib_path, name)
index_list = sorted(os.listdir(root_path))
intrinsic, extrinsic = load_camparam(os.path.join(root_path, "0"))

undistort_and_detect_charuco(name)
compute_fk(name, args.arm)
motion_wrt_cam, motion_wrt_robot = compute_motion(name)
robot_wrt_cam = solve_ax_xb(motion_wrt_cam, motion_wrt_robot, verbose=True)

for i in range(len(motion_wrt_cam)):
    diff = (motion_wrt_cam[i] @ robot_wrt_cam) - (robot_wrt_cam @ motion_wrt_robot[i])
    trans_error = np.linalg.norm(diff[:3, 3]) * 1000

    R_error = diff[:3, :3]
    cos_arg = np.clip((np.trace(R_error.T @ R_error) - 1) / 2, -1.0, 1.0)
    angle_error = np.degrees(np.arccos(cos_arg)) if not np.isnan(cos_arg) else 0.0
    print(f"Motion {i}: trans={trans_error:.2f}mm, rotF={np.linalg.norm(R_error,'fro'):.4f}")

# Joint nonlinear refinement removes the absolute-pose bias that AX=XB cannot see
robot_wrt_cam = refine_handeye_joint(name, robot_wrt_cam)
cam_wrt_robot = np.linalg.inv(robot_wrt_cam)

np.save(os.path.join(root_path, index_list[0], "C2R.npy"), robot_wrt_cam)

debug(name, args.arm)