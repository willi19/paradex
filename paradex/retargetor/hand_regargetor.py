import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

import time
import copy


_MEDIAPIPE_TO_MANUS = (
    1, 22, 23, 24, 25,
    3, 4, 5, 6,
    8, 9, 10, 11,
    13, 14, 15, 16,
    18, 19, 20, 21,
)

_WUJI_RETARGETERS = {}
_WUJI_JOINT_LIMITS = {}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_WUJI_THIRDPARTY_ROOT = _REPO_ROOT / "thirdparty" / "wuji-retargeting"
_WUJI_DIRECT_FINGERS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
_WUJI_DIRECT_MCP_FROM_PIP_BLEND = 0.55
_WUJI_DIRECT_MCP_FLEX_GAIN = 1.15
_WUJI_DIRECT_THUMB_OPPOSITION_GAIN = 1.9
_WUJI_DIRECT_FALLBACK_LOWER = np.array(
    [
        0.0475, -0.1387, -0.4642, -0.4699,
        -0.1585, -0.3700, -0.4777, -0.4683,
        -0.1644, -0.3700, -0.4739, -0.4684,
        -0.1554, -0.3700, -0.4765, -0.4777,
        -0.1626, -0.3700, -0.4768, -0.4683,
    ],
    dtype=np.float32,
)
_WUJI_DIRECT_FALLBACK_UPPER = np.array(
    [
        1.6033, 0.9324, 1.5623, 1.5568,
        1.5604, 0.3700, 1.5485, 1.5753,
        1.5516, 0.3700, 1.5512, 1.5745,
        1.5585, 0.3700, 1.5487, 1.5634,
        1.5585, 0.3700, 1.5490, 1.5735,
    ],
    dtype=np.float32,
)


def _pose_position(pose):
    if isinstance(pose, np.ndarray):
        pose = np.asarray(pose)
        if pose.shape == (4, 4):
            return pose[:3, 3].astype(np.float32)
        if pose.shape == (3,):
            return pose.astype(np.float32)

    if hasattr(pose, "position"):
        p = pose.position
        return np.array([p.x, p.y, p.z], dtype=np.float32)

    if isinstance(pose, dict):
        if "pose" in pose:
            return _pose_position(pose["pose"])
        if "position" in pose:
            return _pose_position(pose["position"])
        if {"x", "y", "z"}.issubset(pose):
            return np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float32)

    return None


def _manus_position(pos):
    # The standalone Wuji Manus ROS2 path mirrors raw Manus Y before retargeting.
    pos = np.asarray(pos, dtype=np.float32)
    return np.array([pos[0], -pos[1], pos[2]], dtype=np.float32)


def _named_pose_position(pos):
    # Xsens/named poses are already in the receiver's corrected global frame.
    return np.asarray(pos, dtype=np.float32)


def _node_id(node):
    if hasattr(node, "node_id"):
        return int(node.node_id)
    if isinstance(node, dict):
        if "node_id" in node:
            return int(node["node_id"])
        if "id" in node:
            return int(node["id"])
    return None


def _node_pose(node):
    if hasattr(node, "pose"):
        return node.pose
    if isinstance(node, dict):
        return node.get("pose", node)
    return node


def _raw_nodes_from_frame(hand_pose_frame):
    if hasattr(hand_pose_frame, "raw_nodes"):
        return hand_pose_frame.raw_nodes
    if isinstance(hand_pose_frame, (list, tuple)):
        return hand_pose_frame
    if isinstance(hand_pose_frame, dict):
        if "raw_nodes" in hand_pose_frame:
            return hand_pose_frame["raw_nodes"]
        if hand_pose_frame and all(isinstance(k, (int, np.integer)) for k in hand_pose_frame):
            return [
                {"node_id": int(node_id), "pose": pose}
                for node_id, pose in hand_pose_frame.items()
            ]
    return None


def _manus_to_mediapipe(hand_pose_frame):
    raw_nodes = _raw_nodes_from_frame(hand_pose_frame)
    if raw_nodes is None:
        return None

    positions = {}
    for node in raw_nodes:
        node_id = _node_id(node)
        pos = _pose_position(_node_pose(node))
        if node_id is not None and pos is not None:
            positions[node_id] = _manus_position(pos)

    if 1 not in positions:
        return None

    if any(manus_id not in positions for manus_id in _MEDIAPIPE_TO_MANUS):
        return None

    keypoints = np.zeros((21, 3), dtype=np.float32)
    for mp_idx, manus_id in enumerate(_MEDIAPIPE_TO_MANUS):
        keypoints[mp_idx] = positions[manus_id]

    return keypoints


def _named_pose_point(hand_pose_frame, name):
    if name not in hand_pose_frame:
        return None
    pos = _pose_position(hand_pose_frame[name])
    if pos is None:
        return None
    return _named_pose_position(pos)


def _finger_points(hand_pose_frame, names, count=4):
    points = []
    for name in names:
        point = _named_pose_point(hand_pose_frame, name)
        if point is not None:
            points.append(point)
    if not points:
        return None

    while len(points) < count:
        if len(points) >= 2:
            points.append(points[-1] + (points[-1] - points[-2]))
        else:
            points.append(points[-1].copy())

    return points[:count]


def _named_hand_pose_to_mediapipe(hand_pose_frame):
    if not isinstance(hand_pose_frame, dict) or "wrist" not in hand_pose_frame:
        return None

    wrist = _named_pose_point(hand_pose_frame, "wrist")
    if wrist is None:
        return None

    keypoints = np.zeros((21, 3), dtype=np.float32)
    keypoints[0] = wrist

    finger_specs = [
        (1, ["thumb_metacarpal", "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip"]),
        (5, ["index_metacarpal", "index_proximal", "index_intermediate", "index_distal"]),
        (9, ["middle_metacarpal", "middle_proximal", "middle_intermediate", "middle_distal"]),
        (13, ["ring_metacarpal", "ring_proximal", "ring_intermediate", "ring_distal"]),
        (17, ["pinky_metacarpal", "pinky_proximal", "pinky_intermediate", "pinky_distal"]),
    ]

    for start_idx, names in finger_specs:
        points = _finger_points(hand_pose_frame, names, count=4)
        if points is None:
            return None
        keypoints[start_idx:start_idx + 4] = points

    return keypoints


def _to_wuji_mediapipe(hand_pose_frame):
    arr = np.asarray(hand_pose_frame) if not isinstance(hand_pose_frame, dict) else None
    if arr is not None:
        if arr.shape == (21, 3):
            return arr.astype(np.float32)
        if arr.shape == (63,):
            return arr.reshape(21, 3).astype(np.float32)

    keypoints = _manus_to_mediapipe(hand_pose_frame)
    if keypoints is not None:
        return keypoints

    return _named_hand_pose_to_mediapipe(hand_pose_frame)


def _scale_keypoints_about_wrist(keypoints, scale):
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Wuji hand scale must be a positive finite number, got {scale}")
    if scale == 1.0:
        return keypoints
    wrist = keypoints[0:1]
    return wrist + (keypoints - wrist) * scale


def _unit_vector(vec, fallback=None):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-6 or not np.isfinite(norm):
        if fallback is None:
            return None
        return np.asarray(fallback, dtype=np.float32)
    return vec / norm


def _project_to_plane(vec, normal):
    normal = _unit_vector(normal)
    if normal is None:
        return np.asarray(vec, dtype=np.float32)
    vec = np.asarray(vec, dtype=np.float32)
    return vec - np.dot(vec, normal) * normal


def _angle_between(vec_a, vec_b):
    a = _unit_vector(vec_a)
    b = _unit_vector(vec_b)
    if a is None or b is None:
        return 0.0
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def _signed_angle_between(vec_a, vec_b, axis):
    a = _unit_vector(vec_a)
    b = _unit_vector(vec_b)
    axis = _unit_vector(axis)
    if a is None or b is None or axis is None:
        return 0.0
    unsigned = _angle_between(a, b)
    sign = np.sign(np.dot(np.cross(a, b), axis))
    if sign == 0.0:
        sign = 1.0
    return float(unsigned * sign)


def _wuji_direct_palm_frame(keypoints):
    wrist = keypoints[0]
    index_mcp = keypoints[5]
    middle_mcp = keypoints[9]
    pinky_mcp = keypoints[17]

    lateral = _unit_vector(index_mcp - pinky_mcp, fallback=np.array([1.0, 0.0, 0.0]))
    forward = _unit_vector(middle_mcp - wrist, fallback=np.array([0.0, 1.0, 0.0]))
    normal = _unit_vector(np.cross(lateral, forward), fallback=np.array([0.0, 0.0, 1.0]))
    forward = _unit_vector(np.cross(normal, lateral), fallback=forward)
    return lateral, forward, normal


def _resolve_wuji_direct_limits(is_right=True):
    side = "right" if is_right else "left"
    if side in _WUJI_JOINT_LIMITS:
        return _WUJI_JOINT_LIMITS[side]

    lower = []
    upper = []
    urdf_path = (
        _WUJI_THIRDPARTY_ROOT
        / "wuji_retargeting"
        / "wuji-description"
        / "hand"
        / "body"
        / "urdf"
        / f"{side}.urdf"
    )
    try:
        import xml.etree.ElementTree as ET

        root = ET.parse(urdf_path).getroot()
        for joint in root.findall("joint"):
            if joint.get("type") == "fixed":
                continue
            limit = joint.find("limit")
            if limit is None:
                continue
            lower.append(float(limit.get("lower")))
            upper.append(float(limit.get("upper")))
    except Exception:
        lower = []
        upper = []

    if len(lower) != 20 or len(upper) != 20:
        limits = (_WUJI_DIRECT_FALLBACK_LOWER, _WUJI_DIRECT_FALLBACK_UPPER)
    else:
        limits = (
            np.asarray(lower, dtype=np.float32),
            np.asarray(upper, dtype=np.float32),
        )
    _WUJI_JOINT_LIMITS[side] = limits
    return limits


def _wuji_direct_non_thumb_angles(keypoints, finger_indices, normal):
    mcp, pip, dip, tip = [keypoints[i] for i in finger_indices]
    base_dir = _project_to_plane(mcp - keypoints[0], normal)
    prox = pip - mcp
    mid = dip - pip
    distal = tip - dip

    base_dir = _unit_vector(base_dir, fallback=prox)
    side_axis = _unit_vector(np.cross(normal, base_dir), fallback=np.array([1.0, 0.0, 0.0]))
    prox_in_flex_plane = prox - np.dot(prox, side_axis) * side_axis

    abduction = _signed_angle_between(
        _project_to_plane(base_dir, normal),
        _project_to_plane(prox, normal),
        normal,
    )
    pip_flex = _angle_between(prox, mid)
    dip_flex = _angle_between(mid, distal)
    geometric_mcp_flex = _angle_between(base_dir, prox_in_flex_plane)
    mcp_flex = max(geometric_mcp_flex, _WUJI_DIRECT_MCP_FROM_PIP_BLEND * pip_flex)
    mcp_flex *= _WUJI_DIRECT_MCP_FLEX_GAIN
    return np.array([mcp_flex, abduction, pip_flex, dip_flex], dtype=np.float32)


def _wuji_direct_thumb_angles(keypoints, forward, normal):
    cmc, mcp, ip, tip = [keypoints[i] for i in _WUJI_DIRECT_FINGERS[0]]
    wrist = keypoints[0]
    metacarpal = mcp - cmc
    proximal = ip - mcp
    distal = tip - ip

    thumb_plane = _project_to_plane(metacarpal, normal)
    opposition = abs(_signed_angle_between(forward, thumb_plane, normal))
    cmc_flex = _angle_between(cmc - wrist, metacarpal)
    mcp_flex = _angle_between(metacarpal, proximal)
    ip_flex = _angle_between(proximal, distal)
    opposition *= _WUJI_DIRECT_THUMB_OPPOSITION_GAIN
    return np.array([cmc_flex, opposition, mcp_flex, ip_flex], dtype=np.float32)


def _wuji_direct_from_mediapipe(keypoints, is_right=True):
    if keypoints.shape != (21, 3) or not np.all(np.isfinite(keypoints)):
        return None

    _, forward, normal = _wuji_direct_palm_frame(keypoints)
    angles = np.zeros((5, 4), dtype=np.float32)
    angles[0] = _wuji_direct_thumb_angles(keypoints, forward, normal)
    for finger_id, finger_indices in enumerate(_WUJI_DIRECT_FINGERS[1:], start=1):
        angles[finger_id] = _wuji_direct_non_thumb_angles(keypoints, finger_indices, normal)

    if not is_right:
        angles[1:, 1] *= -1.0

    lower, upper = _resolve_wuji_direct_limits(is_right=is_right)
    return np.clip(angles.reshape(20), lower, upper).astype(np.float32)


def _import_wuji_retargeter():
    candidate_paths = []
    env_path = os.environ.get("WUJI_RETARGETING_PATH")
    if env_path:
        candidate_paths.append(Path(env_path))
    candidate_paths.append(_WUJI_THIRDPARTY_ROOT)

    for path in reversed(candidate_paths):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    try:
        from wuji_retargeting import Retargeter
        return Retargeter
    except ImportError as exc:
        raise RuntimeError(
            "wuji hand retargeting requires wuji_retargeting and its dependencies. "
            "Use thirdparty/wuji-retargeting, set WUJI_RETARGETING_PATH if needed, "
            f"and make sure optimizer dependencies are available: {exc}"
        ) from exc


def _resolve_wuji_config_path(side):
    side = side.lower()
    direct_env = os.environ.get(f"WUJI_RETARGET_CONFIG_{side.upper()}") or os.environ.get("WUJI_RETARGET_CONFIG")
    if direct_env:
        path = Path(direct_env)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Wuji retarget config not found: {path}")

    candidate_dirs = []
    config_dir = os.environ.get("WUJI_RETARGET_CONFIG_DIR")
    if config_dir:
        candidate_dirs.append(Path(config_dir))

    env_path = os.environ.get("WUJI_RETARGETING_PATH")
    if env_path:
        candidate_dirs.append(Path(env_path) / "example" / "config")

    candidate_dirs.append(_WUJI_THIRDPARTY_ROOT / "example" / "config")

    for cfg_dir in candidate_dirs:
        path = cfg_dir / f"retarget_manus_{side}.yaml"
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"Could not find retarget_manus_{side}.yaml. "
        "Set WUJI_RETARGET_CONFIG_DIR or WUJI_RETARGET_CONFIG."
    )


def _get_wuji_retargeter(is_right=True):
    side = "right" if is_right else "left"
    config_path = _resolve_wuji_config_path(side)
    key = (side, config_path)

    if key not in _WUJI_RETARGETERS:
        Retargeter = _import_wuji_retargeter()
        _WUJI_RETARGETERS[key] = Retargeter.from_yaml(config_path, side)

    return _WUJI_RETARGETERS[key]


def wuji(hand_pose_frame, is_right=True, scale=1.0):
    keypoints = _to_wuji_mediapipe(hand_pose_frame)
    if keypoints is None:
        return None
    keypoints = _scale_keypoints_about_wrist(keypoints, scale)

    retargeter = _get_wuji_retargeter(is_right=is_right)
    return np.asarray(retargeter.retarget(keypoints), dtype=np.float32)


def wuji_direct(hand_pose_frame, is_right=True, scale=1.0):
    keypoints = _to_wuji_mediapipe(hand_pose_frame)
    if keypoints is None:
        return None
    keypoints = _scale_keypoints_about_wrist(keypoints, scale)
    return _wuji_direct_from_mediapipe(keypoints, is_right=is_right)


def wuji_hybrid(hand_pose_frame, is_right=True, scale=1.0):
    keypoints = _to_wuji_mediapipe(hand_pose_frame)
    if keypoints is None:
        return None
    keypoints = _scale_keypoints_about_wrist(keypoints, scale)

    direct_action = _wuji_direct_from_mediapipe(keypoints, is_right=is_right)
    if direct_action is None:
        return None

    retargeter = _get_wuji_retargeter(is_right=is_right)
    opt_action = np.asarray(retargeter.retarget(keypoints), dtype=np.float32)
    if opt_action.shape != (20,) or not np.all(np.isfinite(opt_action)):
        return direct_action

    action = direct_action.copy()
    action[:4] = opt_action[:4]
    return action


def allegro(hand_pose_frame):
    hand_joint_angle = np.zeros((20,3))
    allegro_angles = np.zeros(16)
    # for finger_id in range(4):
    #     for joint_id in range(4):
    #         if joint_id == 0:
    #             rot_mat = np.linalg.inv(hand_pose_frame[0,:3,:3]) @ hand_pose_frame[finger_id * 4 + joint_id + 1, :3,:3]
    #         else:
    #             rot_mat = np.linalg.inv(hand_pose_frame[hand_index.hand_index_parent[finger_id * 4 + joint_id+1], :3,:3]) @ hand_pose_frame[finger_id * 4 + joint_id + 1, :3,:3]
    #         hand_joint_angle[finger_id * 4 + joint_id + 1] = Rotation.from_matrix(rot_mat).as_euler("zyx")
    
    # zyx euler angle in hand frame = zxy axis angle in robot frame
    
    # Ring
    joint_name_list = ["metacarpal", "proximal", "intermediate","distal"]
    for i, finger_name in enumerate(["index", "middle", "ring"]):
        metacarpal = finger_name + "_metacarpal"
        distal = finger_name + "_distal"
        
        tip_position = (np.linalg.inv(hand_pose_frame["wrist"]) @ hand_pose_frame[distal])[:3, 3]
        finger_base_position = (np.linalg.inv(hand_pose_frame["wrist"]) @ hand_pose_frame[metacarpal])[:3, 3]
        
        tip_position = tip_position - finger_base_position
        tip_direction  = tip_position / np.linalg.norm(tip_position)

        if tip_direction[1] > 0.9:
            allegro_angles[4*i] = 0
        else:
            allegro_angles[4*i] = np.arctan(tip_direction[0] / tip_direction[2]) * (0.9-tip_direction[1])
        
        for j in range(3):
            parent_name = finger_name + "_" + joint_name_list[j]
            joint_name = finger_name + "_" + joint_name_list[j+1]
            rot_mat = np.linalg.inv(hand_pose_frame[parent_name][:3,:3]) @ hand_pose_frame[joint_name][:3,:3]
            v = rot_mat[1, 1] if rot_mat[2, 1] >= 0 else 1
            v = max(-1, min(1, v))
            allegro_angles[4*i+j+1] = np.arccos(v)
        allegro_angles[4*i+1] = (allegro_angles[4*i+1]-0.35) * 1.5


    # Thumb
    thumb_meta = np.dot(hand_pose_frame["wrist"][:3,:3].T, hand_pose_frame["thumb_metacarpal"][:3,:3])
    thumb_meta_angle = R.from_matrix(thumb_meta).as_euler("xyz")
    allegro_angles[12] = thumb_meta_angle[0] 
    allegro_angles[13] = -thumb_meta_angle[2]-1.57

    for i, (parent_name, joint_name) in enumerate([("thumb_metacarpal", "thumb_proximal"),("thumb_proximal", "thumb_distal")]):
        rot_mat = np.linalg.inv(hand_pose_frame[parent_name][:3,:3]) @ hand_pose_frame[joint_name][:3,:3]
        allegro_angles[14+i] = rot_mat[2, 1] * 1.2

    return allegro_angles

def inspire(hand_pose_frame):
    inspire_angles = np.zeros(6)

    for i, finger_name in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
        metacarpal = finger_name + "_metacarpal"
        distal = finger_name + "_distal"
        
        tip_pos = (np.linalg.inv(hand_pose_frame["wrist"]) @ hand_pose_frame[distal])
        
        if finger_name != "thumb":
            angle = np.arctan2(tip_pos[2, 1], tip_pos[1, 1])
            if angle < -np.pi / 2:
                angle = 2 * np.pi + angle
            
            inspire_angles[4-i] = (1-max(0, min(1, angle / np.pi))) * 1000

        else:
            tip_position = tip_pos[:3, 3]
            finger_base_position = (np.linalg.inv(hand_pose_frame["wrist"]) @ hand_pose_frame[metacarpal])[:3, 3]
            tip_direction = tip_position - finger_base_position
            tip_direction  = tip_direction / np.linalg.norm(tip_direction)
            tip_direction[1] *= -1
            tip_direction[2] *= -1
            if tip_direction[0] > 0:
                inspire_angles[5] = 1000 - np.arctan(-tip_direction[2] / abs(tip_direction[0])) / np.pi * 2000
                inspire_angles[4] = np.arccos(-tip_direction[1]) * 2000 - 1000 # no divide by pi for better range
            else:
                inspire_angles[5] = 0
                inspire_angles[4] = np.arcsin(-tip_direction[2]) / np.pi * 2000  * 3.8 - 1000
    # print(inspire_angles)
    return inspire_angles

def inspire_f1_deprecated(hand_pose_frame):
    """
    Same kinematic mapping as inspire(), but scale each DOF to Inspire F1 raw range.
    """
    # inspire_angles = inspire(hand_pose_frame)
    # f1_ranges = np.array([
    #     [900, 1740],
    #     [900, 1740],
    #     [900, 1740],
    #     [900, 1740],
    #     [1100, 1350],
    #     [600, 1800],
    # ], dtype=np.float64)

    # # Clamp to [0, 1000] then scale to per-DOF range
    # inspire_angles = np.clip(inspire_angles, 0.0, 1000.0)
    # mins = f1_ranges[:, 0]
    # maxs = f1_ranges[:, 1]
    # scaled = mins + (inspire_angles / 1000.0) * (maxs - mins)
    # inspire_angles[:4] = [1000.0] * 4

    
    
    
    inspire_angles = np.zeros(6)

    for i, finger_name in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
        metacarpal = finger_name + "_metacarpal"
        distal = finger_name + "_distal"
        
        tip_pos = (np.linalg.inv(hand_pose_frame["wrist"]) @ hand_pose_frame[distal])
        
        if finger_name != "thumb":
            angle = np.arctan2(tip_pos[2, 1], tip_pos[1, 1])
            if angle < -np.pi / 2:
                angle = 2 * np.pi + angle
            
            inspire_angles[4-i] = (1-max(0, min(1, angle / np.pi))) * 500 + 500

        else:
            tip_position = tip_pos[:3, 3]
            finger_base_position = (np.linalg.inv(hand_pose_frame["wrist"]) @ hand_pose_frame[metacarpal])[:3, 3]
            tip_direction = tip_position - finger_base_position
            tip_direction  = tip_direction / np.linalg.norm(tip_direction)
            tip_direction[1] *= -1
            tip_direction[2] *= -1
            
            print(tip_direction)
            if tip_direction[0] > 0:
                inspire_angles[4] = 1000 - np.arctan(-tip_direction[2] / abs(tip_direction[0])) / np.pi * 250
                inspire_angles[5] = -np.arccos(tip_direction[0]) * 800 + 1500 # no divide by pi for better range
            else:
                inspire_angles[4] = 1000 - np.arctan(-tip_direction[2] / abs(tip_direction[0])) / np.pi * 250
                # inspire_angles[4] = 1000 - np.arctan(-tip_direction[2] / abs(tip_direction[0])) / np.pi * 2000
                print("asfd")
                inspire_angles[5] = 300
    print(inspire_angles)

                
        # inspire_angles[:4] = 500.0 + inspire_angles[:4] * 0.5
        
        
    return inspire_angles
        


def inspire_f1(hand_pose_frame: Dict[str, np.ndarray], is_right: bool = True):
    required = [
        "wrist",
        "thumb_metacarpal",
        "thumb_distal",
        "index_distal",
        "middle_distal",
        "ring_distal",
        "pinky_distal",
    ]
    if any(k not in hand_pose_frame for k in required):
        return None

    inspire_angles = np.zeros(6, dtype=np.float64)
    wrist_inv = np.linalg.inv(hand_pose_frame["wrist"])

    for i, finger_name in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
        metacarpal = f"{finger_name}_metacarpal"
        distal = f"{finger_name}_distal"

        tip_pos = wrist_inv @ hand_pose_frame[distal]

        if finger_name != "thumb":
            angle = np.arctan2(tip_pos[2, 1], tip_pos[1, 1])
            if angle < -np.pi / 2:
                angle = 2 * np.pi + angle
            inspire_angles[4 - i] = (1 - max(0.0, min(1.0, angle / np.pi))) * 840.0 + 900.0
        else:
            tip_position = tip_pos[:3, 3]
            finger_base_position = (wrist_inv @ hand_pose_frame[metacarpal])[:3, 3]
            tip_direction = tip_position - finger_base_position
            norm = np.linalg.norm(tip_direction)
            if norm < 1e-8:
                return None
            tip_direction = tip_direction / norm

            if is_right:
                inspire_angles[5] = 660 * tip_direction[0] + 700
                if tip_direction[0] < 0:
                    inspire_angles[4] = 165 * np.arctan(tip_direction[2] / abs(tip_direction[0])) + 968
                else:
                    inspire_angles[4] = 1350
            else:
                inspire_angles[5] = -500 * tip_direction[0] + 850
                if tip_direction[0] > 0:
                    inspire_angles[4] = -125 * np.arctan(tip_direction[2] / abs(tip_direction[0])) + 1025
                else:
                    inspire_angles[4] = 1350

    inspire_angles = np.clip(np.rint(inspire_angles), 0, 1740).astype(np.int32)
    inspire_angles[5] = np.clip(inspire_angles[5], 600, 1800)
    inspire_angles[4] = np.clip(inspire_angles[4], 1100, 1350)

    return inspire_angles


def robotiq_2f85(hand_pose_frame: Dict[str, np.ndarray]):
    required = ["thumb_distal", "index_distal"]
    if any(k not in hand_pose_frame for k in required):
        return None

    thumb_tip = hand_pose_frame["thumb_distal"][:3, 3]
    index_tip = hand_pose_frame["index_distal"][:3, 3]
    pinch_distance = np.linalg.norm(thumb_tip - index_tip)
    if not np.isfinite(pinch_distance):
        return None

    closed_distance = 0.025
    open_distance = 0.100
    close_value = (open_distance - pinch_distance) / (open_distance - closed_distance)
    close_value = float(np.clip(close_value, 0.0, 1.0))

    return np.asarray([close_value], dtype=np.float64)



def kistar(hand_pose_frame):
    required = [
        "wrist",
        "thumb_metacarpal",
        "thumb_proximal",
        "thumb_distal",
        "index_metacarpal",
        "index_proximal",
        "index_intermediate",
        "index_distal",
        "middle_metacarpal",
        "middle_proximal",
        "middle_intermediate",
        "middle_distal",
        "ring_metacarpal",
        "ring_proximal",
        "ring_intermediate",
        "ring_distal",
    ]
    if any(k not in hand_pose_frame for k in required):
        return None

    def _angle_to_raw(v, max_angle=1.8):
        v = float(np.clip(v, 0.0, max_angle))
        return int(np.clip(np.rint(v / max_angle * 4096.0), 0, 4096))

    # intermediate representation in radians (allegro-like extraction)
    allegro_angles = np.zeros(16, dtype=np.float64)
    joint_name_list = ["metacarpal", "proximal", "intermediate", "distal"]
    wrist_inv = np.linalg.inv(hand_pose_frame["wrist"])

    for i, finger_name in enumerate(["index", "middle", "ring"]):
        metacarpal = f"{finger_name}_metacarpal"
        distal = f"{finger_name}_distal"

        tip_position = (wrist_inv @ hand_pose_frame[distal])[:3, 3]
        finger_base_position = (wrist_inv @ hand_pose_frame[metacarpal])[:3, 3]
        tip_direction = tip_position - finger_base_position
        norm = np.linalg.norm(tip_direction)
        if norm < 1e-8:
            return None
        tip_direction = tip_direction / norm

        if tip_direction[1] > 0.9:
            allegro_angles[4 * i] = 0.0
        else:
            allegro_angles[4 * i] = np.arctan2(tip_direction[0], tip_direction[2]) * (0.9 - tip_direction[1])

        for j in range(3):
            parent_name = f"{finger_name}_{joint_name_list[j]}"
            joint_name = f"{finger_name}_{joint_name_list[j + 1]}"
            rot_mat = np.linalg.inv(hand_pose_frame[parent_name][:3, :3]) @ hand_pose_frame[joint_name][:3, :3]
            v = rot_mat[1, 1] if rot_mat[2, 1] >= 0 else 1
            v = max(-1.0, min(1.0, v))
            allegro_angles[4 * i + j + 1] = np.arccos(v)
        allegro_angles[4 * i + 1] = (allegro_angles[4 * i + 1] - 0.35) * 1.5

    thumb_meta = hand_pose_frame["wrist"][:3, :3].T @ hand_pose_frame["thumb_metacarpal"][:3, :3]
    thumb_meta_angle = R.from_matrix(thumb_meta).as_euler("xyz")
    allegro_angles[12] = thumb_meta_angle[0]
    allegro_angles[13] = -thumb_meta_angle[2] - 1.57
    for i, (parent_name, joint_name) in enumerate(
        [("thumb_metacarpal", "thumb_proximal"), ("thumb_proximal", "thumb_distal")]
    ):
        rot_mat = np.linalg.inv(hand_pose_frame[parent_name][:3, :3]) @ hand_pose_frame[joint_name][:3, :3]
        allegro_angles[14 + i] = rot_mat[2, 1] * 1.2

    # KISTAR raw command (0=open, 4096=closed), 16-dof with fixed indices
    kistar_raw = np.zeros(16, dtype=np.int32)

    # thumb: active [0,2,3], fixed [1]
    kistar_raw[0] = _angle_to_raw(allegro_angles[13], max_angle=0.35)    
    kistar_raw[1] = -_angle_to_raw(allegro_angles[13], max_angle=0.35)

    kistar_raw[2] = _angle_to_raw(allegro_angles[14], max_angle=1.5)
    kistar_raw[3] = _angle_to_raw(allegro_angles[15], max_angle=1.5)

    # index: active [5,6,7], fixed [4]
    kistar_raw[5] = _angle_to_raw(allegro_angles[1], max_angle=1.8)
    kistar_raw[6] = _angle_to_raw(allegro_angles[2], max_angle=1.8)
    kistar_raw[7] = _angle_to_raw(allegro_angles[3], max_angle=1.8)

    # middle: active [9,10,11], fixed [8]
    kistar_raw[9] = _angle_to_raw(allegro_angles[5], max_angle=1.8)
    kistar_raw[10] = _angle_to_raw(allegro_angles[6], max_angle=1.8)
    kistar_raw[11] = _angle_to_raw(allegro_angles[7], max_angle=1.8)

    # ring: active [13,14,15], fixed [12]
    kistar_raw[13] = _angle_to_raw(allegro_angles[9], max_angle=1.8)
    kistar_raw[14] = _angle_to_raw(allegro_angles[10], max_angle=1.8)
    kistar_raw[15] = _angle_to_raw(allegro_angles[11], max_angle=1.8)

    # fixed joints: 1,4,8,12 are already 0
    return kistar_raw


def allegro_v5(hand_pose_frame):
    hand_joint_angle = np.zeros((20,3))
    allegro_angles = np.zeros(16)
    # for finger_id in range(4):
    #     for joint_id in range(4):
    #         if joint_id == 0:
    #             rot_mat = np.linalg.inv(hand_pose_frame[0,:3,:3]) @ hand_pose_frame[finger_id * 4 + joint_id + 1, :3,:3]
    #         else:
    #             rot_mat = np.linalg.inv(hand_pose_frame[hand_index.hand_index_parent[finger_id * 4 + joint_id+1], :3,:3]) @ hand_pose_frame[finger_id * 4 + joint_id + 1, :3,:3]
    #         hand_joint_angle[finger_id * 4 + joint_id + 1] = Rotation.from_matrix(rot_mat).as_euler("zyx")
    
    # zyx euler angle in hand frame = zxy axis angle in robot frame
    
    # Ring
    joint_name_list = ["metacarpal", "proximal", "intermediate","distal"]
    for i, finger_name in enumerate(["index", "middle", "ring"]):
        metacarpal = finger_name + "_metacarpal"
        distal = finger_name + "_distal"
        
        tip_position = (np.linalg.inv(hand_pose_frame["wrist"]) @ hand_pose_frame[distal])[:3, 3]
        finger_base_position = (np.linalg.inv(hand_pose_frame["wrist"]) @ hand_pose_frame[metacarpal])[:3, 3]
        
        tip_position = tip_position - finger_base_position
        tip_direction  = tip_position / np.linalg.norm(tip_position)

        if tip_direction[1] > 0.9:
            allegro_angles[4*i] = 0
        else:
            allegro_angles[4*i] = np.arctan(tip_direction[0] / tip_direction[2]) * (0.9-tip_direction[1])
        
        for j in range(3):
            parent_name = finger_name + "_" + joint_name_list[j]
            joint_name = finger_name + "_" + joint_name_list[j+1]
            rot_mat = np.linalg.inv(hand_pose_frame[parent_name][:3,:3]) @ hand_pose_frame[joint_name][:3,:3]
            v = rot_mat[1, 1] if rot_mat[2, 1] >= 0 else 1
            v = max(-1, min(1, v))
            allegro_angles[4*i+j+1] = np.arccos(v)
        allegro_angles[4*i+1] = (allegro_angles[4*i+1]-0.35) * 1.5


    # Thumb
    thumb_meta = np.dot(hand_pose_frame["wrist"][:3,:3].T, hand_pose_frame["thumb_metacarpal"][:3,:3])
    thumb_meta_angle = R.from_matrix(thumb_meta).as_euler("xyz")
    allegro_angles[12] = thumb_meta_angle[0] 
    allegro_angles[13] = -thumb_meta_angle[2]-1.57

    for i, (parent_name, joint_name) in enumerate([("thumb_metacarpal", "thumb_proximal"),("thumb_proximal", "thumb_distal")]):
        rot_mat = np.linalg.inv(hand_pose_frame[parent_name][:3,:3]) @ hand_pose_frame[joint_name][:3,:3]
        allegro_angles[14+i] = rot_mat[2, 1] * 1.2

    return allegro_angles
