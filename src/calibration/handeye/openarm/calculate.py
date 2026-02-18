import os
import cv2
import argparse
import numpy as np
import tqdm
import random
from copy import deepcopy
from typing import List, Tuple

from paradex.utils.file_io import find_latest_directory

from paradex.calibration.utils import handeye_calib_path_openarm, load_camparam
from paradex.calibration.Tsai_Lenz import solve_ax_xb, solve_axb_cpu    
from paradex.robot.utils import get_robot_urdf_path
from paradex.robot.robot_wrapper_deprecated import RobotWrapper
from paradex.image.image_dict import ImageDict
from paradex.image.aruco import merge_charuco_detection, find_common_indices, detect_charuco
from paradex.transforms.conversion import SOLVE_XA_B
from paradex.visualization.robot import RobotModule


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {v}")


SOURCE_QPOS_ORDER = [
    "openarm_right_joint1",
    "openarm_right_joint4",
    "openarm_left_joint6",
    "openarm_left_joint5",
    "openarm_left_joint2",
    "openarm_left_joint7",
    "openarm_left_joint4",
    "openarm_right_joint5",
    "openarm_left_joint1",
    "openarm_right_joint6",
    "openarm_right_joint3",
    "openarm_left_joint3",
    "openarm_right_joint7",
    "openarm_right_joint2",
]

TARGET_QPOS_ORDER = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
]

def trim_openarm_qpos(qpos):
    qpos = np.asarray(qpos).reshape(-1)
    if qpos.shape[0] < 16:
        raise ValueError(f"Expected at least 16 qpos values, got {qpos.shape[0]}")
    qpos = qpos[2:16]
    if qpos.shape[0] != len(SOURCE_QPOS_ORDER):
        raise ValueError(f"Expected 14 qpos values after trim, got {qpos.shape[0]}")
    source_map = {joint_name: qpos[idx] for idx, joint_name in enumerate(SOURCE_QPOS_ORDER)}
    return np.array([source_map[joint_name] for joint_name in TARGET_QPOS_ORDER], dtype=qpos.dtype)


def _side_root(name: str, side: str) -> str:
    return os.path.join(handeye_calib_path_openarm, name, side)


def _list_capture_indices(root_dir: str) -> List[str]:
    if not os.path.isdir(root_dir):
        return []
    out = []
    for x in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, x)
        if not os.path.isdir(p):
            continue
        if x == "0":
            continue
        out.append(x)
    return out


def undistort_and_detect_charuco(name, side, intrinsic, extrinsic):
    img_dict = None
    root_dir = _side_root(name, side)
    index_list = _list_capture_indices(root_dir)
    
    for index in tqdm.tqdm(index_list, desc=f"Undistort and detect charuco ({side})"):
        print(f"Processing index {index}...")
        if os.path.exists(os.path.join(root_dir, index, "charuco_3d_ids.npy")) and \
           os.path.exists(os.path.join(root_dir, index, "charuco_3d_corners.npy")):
            continue

        if os.path.exists(os.path.join(root_dir, index, "undistort", "images")) and \
            len(os.listdir(os.path.join(root_dir, index, "undistort", "images"))) == \
            len(os.listdir(os.path.join(root_dir, index, "images"))):
            continue
        
        os.makedirs(os.path.join(root_dir, index, "undistort", "images"), exist_ok=True)
        if img_dict is None:
            img_dict = ImageDict.from_path(os.path.join(root_dir, index))
            img_dict.set_camparam(intrinsic, extrinsic)
        else:
            img_dict.update_path(os.path.join(root_dir, index))
            img_dict.set_camparam(intrinsic, extrinsic)
        
        print(f"Undistorting and detecting charuco for index {index}...")
        # if index=='21' or '22':
        #     print
        undistort_img_dict = img_dict.undistort(save_path=os.path.join(root_dir, index, "undistort"))
        
        charuco_3d = undistort_img_dict.triangulate_charuco()
        charuco_3d = merge_charuco_detection(charuco_3d)
        
        charuco_2d = undistort_img_dict.apply(detect_charuco, False)
        detection = {}
        for serial in charuco_2d:
            detection[serial] = merge_charuco_detection(charuco_2d[serial])['checkerCorner']
        
        detectionDict = undistort_img_dict.draw_keypoint(detection, color=(0,255,0))
        print(charuco_3d['checkerCorner'].shape)
        if len(charuco_3d['checkerCorner'])==0:
            print(f"No charuco corners detected for index {index}, skipping saving detections.")
            continue
        projected_dict = undistort_img_dict.project_pointcloud(charuco_3d['checkerCorner'])
        detectionDict = detectionDict.draw_keypoint(projected_dict, color=(255,0,0))
        detectionDict.save(os.path.join(root_dir, index, "detection"))
        
        np.save(os.path.join(root_dir, index, "charuco_3d_ids.npy"), charuco_3d['checkerIDs'])
        np.save(os.path.join(root_dir, index, "charuco_3d_corners.npy"), charuco_3d['checkerCorner'])

def compute_fk(name, arm, side, eef_link):
    root_dir = _side_root(name, side)
    index_list = _list_capture_indices(root_dir)

    robot_wrapper = RobotWrapper(get_robot_urdf_path(arm_name=arm))
    # print(robot_wrapper.joint_names)
    for index in index_list:
        # if os.path.exists(os.path.join(root_dir, index, "eef_fk.npy")):
        #     continue
        qpos = np.load(os.path.join(root_dir, index, "qpos.npy"))
        # qpos = trim_openarm_qpos(qpos)
        # print(qpos)
        eef = robot_wrapper.compute_forward_kinematics(qpos, link_list=[eef_link])[eef_link]
        np.save(os.path.join(root_dir, index, "eef_fk.npy"), eef)

def get_valid_indices(root_dir):
    """Get indices that have valid charuco detection files with actual points."""
    index_list = sorted(os.listdir(root_dir))
    valid_indices = []
    for index in index_list:
        index_path = os.path.join(root_dir, index)
        if not os.path.isdir(index_path):
            continue
        ids_path = os.path.join(index_path, "charuco_3d_ids.npy")
        corners_path = os.path.join(index_path, "charuco_3d_corners.npy")
        if os.path.exists(ids_path) and os.path.exists(corners_path):
            # Check if files have actual points (not empty)
            ids = np.load(ids_path)
            if len(ids) == 0:
                print(f"Skipping index {index}: empty charuco detection")
                continue
            valid_indices.append(index)
        else:
            print(f"Skipping index {index}: missing charuco detection files")
    valid_indices.sort(key=lambda x: int(x) if x.isdigit() else x)
    return valid_indices

def compute_motion(name, side):
    motion_wrt_cam = []
    motion_wrt_robot = []

    root_dir = _side_root(name, side)
    index_list = get_valid_indices(root_dir)

    if len(index_list) < 2:
        raise ValueError(f"Need at least 2 valid indices for motion computation, got {len(index_list)}")

    eef_list = [np.load(os.path.join(root_dir, index, "eef_fk.npy")) for index in index_list]

    charuco_id_list = [np.load(os.path.join(root_dir, index, "charuco_3d_ids.npy")) for index in index_list]
    charuco_cor_list = [np.load(os.path.join(root_dir, index, "charuco_3d_corners.npy")) for index in index_list]
    
    for i in range(1, len(index_list)):
        print(eef_list[i])
        eef = eef_list[i]
        eef_prev = eef_list[i-1]

        motion_wrt_robot.append(eef_prev @ np.linalg.inv(eef)) #M1 R1 R2 M2
        
        ids = charuco_id_list[i]
        ids_prev = charuco_id_list[i-1]
        
        common_idx, common_idx_prev = find_common_indices(ids, ids_prev)

        cam_cor = charuco_cor_list[i][common_idx]
        cam_cor_prev = charuco_cor_list[i-1][common_idx_prev]
        motion_wrt_cam.append(SOLVE_XA_B(cam_cor, cam_cor_prev)) #M1 C1  C2 M2
        # print(np.linalg.norm((eef_prev - eef)[:3, 3]), np.linalg.norm(np.mean(cam_cor - cam_cor_prev, axis=0)))
        # import pdb; pdb.set_trace()
        err = cam_cor_prev - (motion_wrt_cam[-1][:3, :3] @ cam_cor.T).T - motion_wrt_cam[-1][:3, 3]
        print(f"Motion {i-1}->{i} cam points fitting error: {np.mean(np.linalg.norm(err, axis=1))*1000:.2f} mm")
    
    return motion_wrt_cam, motion_wrt_robot, index_list

def debug(name, arm, side, eef_link, robot_wrt_cam_world):
    root_dir = _side_root(name, side)

    index_list = get_valid_indices(root_dir)

    marker_pos = {}
    
    rm = RobotModule(get_robot_urdf_path(arm_name=arm))
    intrinsic, extrinsic = load_camparam(os.path.join(handeye_calib_path_openarm, name, "0"))
    
    for index in index_list:
        eef = np.load(os.path.join(root_dir, index, "eef_fk.npy"))
        # eef_from_robot = np.load(os.path.join(root_dir, index, "eef.npy"))
        # FK error
        # eef_error = np.linalg.inv(eef) @ eef_from_robot
        # trans_error = np.linalg.norm(eef_error[:3, 3]) * 1000
        
        # # Rotation error (degrees)
        # R_error = eef_error[:3, :3]
        # angle_error = np.arccos((np.trace(R_error) - 1) / 2) * 180 / np.pi
        
        # print(f"fk error {index}: trans={trans_error:.2f}mm, rot={angle_error:.2f}deg")
        
        charuco_3d_cor = np.load(os.path.join(root_dir, index, "charuco_3d_corners.npy"))
        charuco_id_cor = np.load(os.path.join(root_dir, index, "charuco_3d_ids.npy"))
        
        for mid, cor in zip(charuco_id_cor, charuco_3d_cor):
            if mid not in marker_pos:
                marker_pos[mid] = []
            cor_h = np.ones((4,))
            cor_h[:3] = cor
            marker_pos[mid].append(((np.linalg.inv(eef) @ np.linalg.inv(robot_wrt_cam_world) @ cor_h.T).T)[:3])

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
        # qpos = np.load(f"./system/current/hecalib/openarm/temp/{int(index) + 1}_qpos.npy")
        # qpos = trim_openarm_qpos(qpos)
        eef = np.load(os.path.join(root_dir, index, "eef_fk.npy"))
        
        rm.update_cfg(qpos)
        robot_mesh = rm.get_robot_mesh()
        robot_mesh.apply_transform(robot_wrt_cam_world)
        
        overlay_img_dict = img_dict.project_mesh(robot_mesh)
        
        marker_wrt_eef_h = np.ones((marker_wrt_eef.shape[0], 4))
        marker_wrt_eef_h[:,:3] = marker_wrt_eef 

        marker_wrt_cam = (robot_wrt_cam_world @ eef @ marker_wrt_eef_h.T).T[:, :3]
        proj_marker = overlay_img_dict.project_pointcloud(marker_wrt_cam)
        overlay_img_dict.draw_keypoint(proj_marker, (255, 0, 0))
        
        marker_3d  = np.load(os.path.join(root_dir, index, "charuco_3d_corners.npy"))
        proj_marker_3d = overlay_img_dict.project_pointcloud(marker_3d)
        overlay_img_dict.draw_keypoint(proj_marker_3d, (0, 0, 255))
        
        overlay_img_dict.save(os.path.join(root_dir, index, "debug"))


def _process_side(
    name: str, arm: str, side: str, intrinsic, extrinsic
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    eef_link = "openarm_left_link7" if side == "left" else "openarm_right_link7"
    undistort_and_detect_charuco(name, side, intrinsic, extrinsic)
    print(f"[{side}] Undistortion and charuco detection completed.")
    compute_fk(name, arm, side, eef_link)
    print(f"[{side}] Forward kinematics computation completed.")
    motion_wrt_cam, motion_wrt_robot, valid_index_list = compute_motion(name, side)
    return motion_wrt_cam, motion_wrt_robot, valid_index_list


def _rotation_error_deg(Rm: np.ndarray) -> float:
    c = (np.trace(Rm) - 1.0) * 0.5
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _pair_residual(A: np.ndarray, B: np.ndarray, X: np.ndarray) -> Tuple[float, float]:
    left = A @ X
    right = X @ B
    delta = np.linalg.inv(left) @ right
    trans_mm = float(np.linalg.norm(delta[:3, 3]) * 1000.0)
    rot_deg = _rotation_error_deg(delta[:3, :3])
    return trans_mm, rot_deg


def _ransac_inlier_indices(
    A_list: List[np.ndarray],
    B_list: List[np.ndarray],
    iterations: int = 300,
    sample_size: int = 8,
    trans_thresh_mm: float = 15.0,
    rot_thresh_deg: float = 3.0,
    seed: int = 0,
) -> List[int]:
    n = len(A_list)
    if n < 3:
        return list(range(n))
    sample_size = max(3, min(sample_size, n))
    rng = random.Random(seed)

    best_inliers: List[int] = []
    best_score = float("inf")

    all_indices = list(range(n))
    for _ in range(iterations):
        sample_idx = rng.sample(all_indices, sample_size)
        A_s = np.array([A_list[i] for i in sample_idx])
        B_s = np.array([B_list[i] for i in sample_idx])
        try:
            X_cand = solve_axb_cpu(A_s, B_s)
        except Exception:
            continue

        inliers = []
        trans_errors = []
        for i, (A, B) in enumerate(zip(A_list, B_list)):
            t_mm, r_deg = _pair_residual(A, B, X_cand)
            # print(t_mm, r_deg)
            if t_mm <= trans_thresh_mm and r_deg <= rot_thresh_deg:
                inliers.append(i)
                trans_errors.append(t_mm)

        if not inliers:
            continue
        mean_t = float(np.mean(trans_errors))
        # Prefer more inliers, then lower mean translation residual.
        score = -len(inliers) * 1e6 + mean_t
        if (not best_inliers) or (score < best_score):
            best_inliers = inliers
            best_score = score

    if len(best_inliers) < max(3, int(0.2 * n)):
        return list(range(n))
    return best_inliers
                
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default=None, help="Name of the calibration directory.")
parser.add_argument("--arm", type=str, default="openarm", help="Name of the robot arm.")
parser.add_argument(
    "--include-left-hand",
    type=_str2bool,
    nargs="?",
    const=True,
    default=True,
    help="Include left-hand sequence in calibration (default: true). Use '--include-left-hand false' for right-only.",
)
parser.add_argument("--include-right-hand", action="store_true", help="Also solve right-hand sequence and fuse into one C2R.")
parser.add_argument("--skip-debug", action="store_true", help="Skip debug overlay rendering.")
parser.add_argument("--ransac", type=_str2bool, nargs="?", const=True, default=True, help="Use RANSAC-based outlier rejection before final AX=XB solve.")
parser.add_argument("--ransac-iter", type=int, default=1000, help="RANSAC iterations.")
parser.add_argument("--ransac-sample-size", type=int, default=10, help="RANSAC minimal sample size.")
parser.add_argument("--ransac-trans-thresh-mm", type=float, default=4.5, help="RANSAC inlier translation threshold in mm.")
parser.add_argument("--ransac-rot-thresh-deg", type=float, default=3.0, help="RANSAC inlier rotation threshold in degrees.")
parser.add_argument("--ransac-seed", type=int, default=0, help="Random seed for RANSAC sampling.")



args = parser.parse_args()
if args.name is None:
    args.name = find_latest_directory(handeye_calib_path_openarm)

name = args.name
root_path = os.path.join(handeye_calib_path_openarm, name)
intrinsic, extrinsic = load_camparam(os.path.join(root_path, "0"))

side_valid_indices = {}
A_all: List[np.ndarray] = []
B_all: List[np.ndarray] = []

if args.include_left_hand:
    left_A, left_B, left_valid = _process_side(name, args.arm, "left", intrinsic, extrinsic)
    A_all.extend(left_A)
    B_all.extend(left_B)
    side_valid_indices["left"] = left_valid

right_root = _side_root(name, "right")
if args.include_right_hand and os.path.isdir(right_root):
    right_A, right_B, right_valid = _process_side(
        name, args.arm, "right", intrinsic, extrinsic
    )
    A_all.extend(right_A)
    B_all.extend(right_B)
    side_valid_indices["right"] = right_valid

if len(A_all) == 0 or len(B_all) == 0:
    raise ValueError("No valid motion pairs found for handeye calibration.")
if len(A_all) != len(B_all):
    raise ValueError(f"Mismatched motion list lengths: len(A)={len(A_all)}, len(B)={len(B_all)}")

if args.ransac:
    inlier_idx = _ransac_inlier_indices(
        A_all,
        B_all,
        iterations=args.ransac_iter,
        sample_size=args.ransac_sample_size,
        trans_thresh_mm=args.ransac_trans_thresh_mm,
        rot_thresh_deg=args.ransac_rot_thresh_deg,
        seed=args.ransac_seed,
    )
    print(f"RANSAC inliers: {len(inlier_idx)}/{len(A_all)}")
    A_solve = [A_all[i] for i in inlier_idx]
    B_solve = [B_all[i] for i in inlier_idx]
else:
    A_solve = A_all
    B_solve = B_all

print(f"Solving AX=XB with pairs: {len(A_solve)} (from total {len(A_all)})")
robot_wrt_cam_world = solve_ax_xb(A_solve, B_solve, verbose=True)

trans_error_list = []
for i in range(len(A_all)):
    diff = (A_all[i] @ robot_wrt_cam_world) - (robot_wrt_cam_world @ B_all[i])
    trans_error = np.linalg.norm(diff[:3, 3]) * 1000.0
    print(f"[all] Motion {i}: trans={trans_error:.2f}mm")
    trans_error_list.append(trans_error)
    


save_dir = os.path.join(root_path, "0")
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, "C2R.npy"), robot_wrt_cam_world)
print(f"Saved C2R to {os.path.join(save_dir, 'C2R.npy')}")
trans_error_path = os.path.join(save_dir, "trans_error_list.txt")
with open(trans_error_path, "w", encoding="utf-8") as f:
    for i, err_mm in enumerate(trans_error_list):
        f.write(f"{i}\t{float(err_mm):.6f}\n")
print(f"Saved translation error list to {trans_error_path}")

if not args.skip_debug:
    # Keep debug lightweight: render left always, and right when enabled.
    if args.include_left_hand:
        debug(name, args.arm, "left", "openarm_left_link7", robot_wrt_cam_world)
    if args.include_right_hand and os.path.isdir(right_root):
        debug(name, args.arm, "right", "openarm_right_link7", robot_wrt_cam_world)
