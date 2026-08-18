import os
import sys
import json
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
_WUJI_DIRECT_ANCHOR_CONFIG = (
    _REPO_ROOT / "rsc" / "robot" / "wuji" / "wuji_left_direct_retargeter_anchors.json"
)
_WUJI_DIRECT_ANCHOR_DATA = {}
# Wuji is ordered thumb, index, middle, ring, pinky, with four joints each.
# A calibration block is therefore always local to one human finger and its
# matching four Wuji joints.
_WUJI_DIRECT_FINGER_INTERPOLATION_LAYOUT = tuple(
    (slice(finger * 4, (finger + 1) * 4), slice(finger * 3, (finger + 1) * 3))
    for finger in range(5)
)
# MANUS tip coordinates are normalized by palm length while the geometric
# direct extractor is radians. Keep the latter a tie-breaker only, so tip
# motion remains the primary metric.
_WUJI_DIRECT_TIP_RAW_TIE_BREAK_SCALE_PALM_LENGTHS_PER_RAD = 0.04
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

# Allegro v5 action order is index, middle, ring, thumb, with four joints per
# finger.  The seven manually edited captures are the entire output model:
# every command is a finger-local interpolation of their target blocks.  The
# raw direct-angle extractor is used only as a distance-feature tie-breaker;
# it never contributes an output pose or a fallback calibration.
_ALLEGRO_V5_DIRECT_ANCHOR_CONFIG = (
    _REPO_ROOT / "rsc" / "robot" / "allegro_v5" / "allegro_v5_direct_retargeter_anchors.json"
)
# Exact hard limits from allegro_v5/allegro_right_A.urdf in driver order.
_ALLEGRO_V5_PHYSICAL_LOWER = np.array(
    [-0.47, -0.196, -0.174, -0.227] * 3 + [0.0, -0.26, 0.0, -0.09],
    dtype=np.float64,
)
_ALLEGRO_V5_PHYSICAL_UPPER = np.array(
    [0.47, 1.8, 1.709, 1.618] * 3 + [1.78, 1.78, 1.9, 1.8],
    dtype=np.float64,
)
_ALLEGRO_V5_DIRECT_ANCHOR_DATA = None
_ALLEGRO_V5_ERGONOMIC_COMPATIBILITY_BLEND = 0.25
_ALLEGRO_V5_ERGONOMIC_FIELDS = (
    "ThumbMCPStretch",
    "ThumbMCPSpread",
    "IndexMCPStretch",
    "IndexPIPStretch",
    "IndexDIPStretch",
    "MiddleMCPStretch",
    "MiddlePIPStretch",
    "MiddleDIPStretch",
    "RingMCPStretch",
    "RingPIPStretch",
    "RingDIPStretch",
    "PinkyMCPStretch",
    "PinkyPIPStretch",
    "PinkyDIPStretch",
)
# The raw Allegro extractor mirrors the legacy transform path's complete
# 16-axis contract.  Unlike the compact six-value calibration feature above,
# it needs the individual joint angles and MCP spread for every actuated
# MANUS digit.
_ALLEGRO_V5_RAW_ERGONOMIC_FIELDS = (
    "ThumbMCPStretch",
    "ThumbMCPSpread",
    "ThumbPIPStretch",
    "ThumbDIPStretch",
    "IndexSpread",
    "IndexMCPStretch",
    "IndexPIPStretch",
    "IndexDIPStretch",
    "MiddleSpread",
    "MiddleMCPStretch",
    "MiddlePIPStretch",
    "MiddleDIPStretch",
    "RingSpread",
    "RingMCPStretch",
    "RingPIPStretch",
    "RingDIPStretch",
)
# Increase the three thumb axes after the base joint while preserving their
# existing directions.  The legacy distal scale is 1.2, so a 1.5 multiplier
# reaches at most 1.8 rad for joints 14/15.
_ALLEGRO_V5_THUMB_SECONDARY_FLEX_GAIN = 1.5
# Allegro v5 action order is index, middle, ring, thumb.  MANUS tip features
# retain the native thumb, index, middle, ring, pinky order.  There is no
# separate Allegro pinky, so it intentionally has no target block here.
_ALLEGRO_V5_FINGER_INTERPOLATION_LAYOUT = (
    ("index", slice(0, 4), slice(3, 6), slice(2, 3)),
    ("middle", slice(4, 8), slice(6, 9), slice(3, 4)),
    ("ring", slice(8, 12), slice(9, 12), slice(4, 5)),
    ("thumb", slice(12, 16), slice(0, 3), slice(0, 2)),
)
# MANUS tip positions are expressed in metres while raw retargeter actions are
# radians.  The raw action is a small finger-local tie-breaker: it separates
# captures with the same fingertip position but a different local bend without
# allowing radians to dominate the Cartesian tip distance.
_ALLEGRO_V5_TIP_RAW_TIE_BREAK_SCALE_M_PER_RAD = 0.1


def _load_allegro_v5_direct_anchor_data():
    """Load the seven operator-edited anchors for the direct-angle retargeter."""
    global _ALLEGRO_V5_DIRECT_ANCHOR_DATA
    if _ALLEGRO_V5_DIRECT_ANCHOR_DATA is not None:
        return _ALLEGRO_V5_DIRECT_ANCHOR_DATA
    with _ALLEGRO_V5_DIRECT_ANCHOR_CONFIG.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if (
        raw.get("schema_version") != 2
        or raw.get("method")
        != "allegro_v5_manus_anchor_shepard_blend"
    ):
        raise RuntimeError("Unsupported Allegro v5 direct-angle anchor calibration")
    interpolation = raw.get("interpolation", {})
    data = {
        "raw_features": np.asarray(raw["anchor_raw_actions_rad"], dtype=np.float64),
        "manus_tip_features": np.asarray(
            raw["anchor_manus_tip_features_m"], dtype=np.float64
        ),
        "manus_ergonomic_features": np.asarray(
            raw["anchor_manus_ergonomic_features_deg"], dtype=np.float64
        ),
        "targets": np.asarray(raw["anchor_target_actions_rad"], dtype=np.float64),
        "safe_lower": np.asarray(raw["safe_lower_rad"], dtype=np.float64),
        "safe_upper": np.asarray(raw["safe_upper_rad"], dtype=np.float64),
        "sigma": float(interpolation["gaussian_sigma_m"]),
        "influence_radius": float(interpolation["influence_radius_m"]),
        "inverse_distance_epsilon": float(
            interpolation["inverse_distance_epsilon_m_sq"]
        ),
        "inverse_distance_power": float(interpolation["inverse_distance_power"]),
        "ergonomic_sigma": float(raw["ergonomic_interpolation"]["gaussian_sigma_deg"]),
        "ergonomic_influence_radius": float(
            raw["ergonomic_interpolation"]["influence_radius_deg"]
        ),
        "ergonomic_inverse_distance_epsilon": float(
            raw["ergonomic_interpolation"]["inverse_distance_epsilon_deg_sq"]
        ),
        "ergonomic_inverse_distance_power": float(
            raw["ergonomic_interpolation"]["inverse_distance_power"]
        ),
    }
    if (
        data["raw_features"].shape != (7, 16)
        or data["manus_tip_features"].shape != (7, 15)
        or data["manus_ergonomic_features"].shape != (7, 6)
        or data["targets"].shape != (7, 16)
        or data["safe_lower"].shape != (16,)
        or data["safe_upper"].shape != (16,)
        or not np.all(np.isfinite(data["raw_features"]))
        or not np.all(np.isfinite(data["manus_tip_features"]))
        or not np.all(np.isfinite(data["manus_ergonomic_features"]))
        or not np.all(np.isfinite(data["targets"]))
        or not np.all(np.isfinite(data["safe_lower"]))
        or not np.all(np.isfinite(data["safe_upper"]))
        or not np.all(data["safe_lower"] <= data["safe_upper"])
        or not np.all(_ALLEGRO_V5_PHYSICAL_LOWER <= data["safe_lower"])
        or not np.all(data["safe_upper"] <= _ALLEGRO_V5_PHYSICAL_UPPER)
        or not np.isfinite(data["sigma"])
        or data["sigma"] <= 0.0
        or not np.isfinite(data["influence_radius"])
        or data["influence_radius"] <= 0.0
        or not np.isfinite(data["inverse_distance_epsilon"])
        or data["inverse_distance_epsilon"] <= 0.0
        or not np.isfinite(data["inverse_distance_power"])
        or data["inverse_distance_power"] < 1.0
        or not np.isfinite(data["ergonomic_sigma"])
        or data["ergonomic_sigma"] <= 0.0
        or not np.isfinite(data["ergonomic_influence_radius"])
        or data["ergonomic_influence_radius"] <= 0.0
        or not np.isfinite(data["ergonomic_inverse_distance_epsilon"])
        or data["ergonomic_inverse_distance_epsilon"] <= 0.0
        or not np.isfinite(data["ergonomic_inverse_distance_power"])
        or data["ergonomic_inverse_distance_power"] < 1.0
    ):
        raise RuntimeError("Allegro v5 direct-angle anchor calibration is invalid")

    _ALLEGRO_V5_DIRECT_ANCHOR_DATA = data
    return data


def clip_allegro_v5_safe_action(action):
    """Constrain direct v5 commands to the seven approved pose envelope."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (16,):
        raise ValueError(f"Allegro v5 action must have shape (16,), got {action.shape}")
    data = _load_allegro_v5_direct_anchor_data()
    return np.clip(action, data["safe_lower"], data["safe_upper"])


def _allegro_v5_manus_tip_feature(hand_pose_frame):
    """Return wrist-relative MANUS fingertip positions in meters.

    Contact intent is expressed by where the tips are, not just by the local
    bone rotations used by the legacy direct-angle extractor.  This feature is
    invariant to the global hand pose and preserves the original extractor as
    the fallback when a MANUS payload is malformed.
    """
    try:
        wrist = np.asarray(hand_pose_frame["wrist"], dtype=np.float64)
        wrist_inv = np.linalg.inv(wrist)
        feature = np.concatenate(
            [
                (wrist_inv @ np.asarray(hand_pose_frame[name], dtype=np.float64))[
                    :3, 3
                ]
                for name in (
                    "thumb_tip",
                    "index_tip",
                    "middle_tip",
                    "ring_tip",
                    "pinky_tip",
                )
            ]
        )
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return None
    return feature if feature.shape == (15,) and np.all(np.isfinite(feature)) else None


def _allegro_v5_manus_ergonomic_feature(ergonomics):
    """Return a MANUS joint-angle feature without relying on field ordering.

    The first two components preserve thumb bend/opposition.  Each remaining
    component is the total finger flexion, mirroring the semantic quantity
    used by Inspire DFTP while retaining the legacy Allegro pose extractor.
    """
    if not isinstance(ergonomics, dict):
        return None
    try:
        values = {name: float(ergonomics[name]) for name in _ALLEGRO_V5_ERGONOMIC_FIELDS}
    except (KeyError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(tuple(values.values()))):
        return None
    return np.asarray(
        (
            values["ThumbMCPStretch"],
            values["ThumbMCPSpread"],
            sum(values[f"Index{joint}Stretch"] for joint in ("MCP", "PIP", "DIP")),
            sum(values[f"Middle{joint}Stretch"] for joint in ("MCP", "PIP", "DIP")),
            sum(values[f"Ring{joint}Stretch"] for joint in ("MCP", "PIP", "DIP")),
            sum(values[f"Pinky{joint}Stretch"] for joint in ("MCP", "PIP", "DIP")),
        ),
        dtype=np.float64,
    )


def _distance_gated_weights(features, query, *, sigma, epsilon, distance_power):
    """Return normalized local anchor weights, favoring the closest pose."""
    feature_deltas = features - query[None, :]
    distances_sq = np.einsum("ij,ij->i", feature_deltas, feature_deltas)
    log_weights = (
        -0.5 * distances_sq / sigma ** 2
        - distance_power * np.log(distances_sq + epsilon)
    )
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    return weights / weights.sum(), distances_sq


def _distance_gated_target(
    features, query, targets, *, sigma, influence_radius, epsilon, distance_power
):
    """Blend a target block from its matching per-anchor feature block."""
    weights, distances_sq = _distance_gated_weights(
        features,
        query,
        sigma=sigma,
        epsilon=epsilon,
        distance_power=distance_power,
    )
    target = weights @ targets
    normalized_distance = np.clip(
        1.0 - np.sqrt(distances_sq.min()) / influence_radius, 0.0, 1.0
    )
    influence = normalized_distance ** 2 * (3.0 - 2.0 * normalized_distance)
    return target, influence


def _cluster_balanced_distance_gated_target(
    features,
    query,
    targets,
    *,
    sigma,
    influence_radius,
    epsilon,
    distance_power,
    duplicate_cluster_radius,
):
    """Blend anchors by unique local pose, not by repeated capture count.

    Nearby duplicate samples are condensed into one centroid/mean-target
    anchor before Shepard weighting.  Thus five captures at one local finger
    pose get the same total influence as one capture at another local pose.
    """
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets):
        raise ValueError("Anchor features and targets must be matching 2D arrays")

    remaining = set(range(len(features)))
    groups = []
    while remaining:
        group = {remaining.pop()}
        frontier = list(group)
        while frontier:
            index = frontier.pop()
            candidates = [
                candidate
                for candidate in remaining
                if np.linalg.norm(features[index] - features[candidate])
                <= duplicate_cluster_radius
            ]
            for candidate in candidates:
                remaining.remove(candidate)
                group.add(candidate)
                frontier.append(candidate)
        groups.append(sorted(group))

    cluster_features = np.asarray(
        [features[group].mean(axis=0) for group in groups], dtype=np.float64
    )
    cluster_targets = np.asarray(
        [targets[group].mean(axis=0) for group in groups], dtype=np.float64
    )
    return _distance_gated_target(
        cluster_features,
        query,
        cluster_targets,
        sigma=sigma,
        influence_radius=influence_radius,
        epsilon=epsilon,
        distance_power=distance_power,
    )


def _align_allegro_v5_action(
    action,
    manus_tip_feature=None,
    manus_ergonomic_feature=None,
):
    """Return a finger-local interpolation of the seven edited target poses."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (16,) or not np.all(np.isfinite(action)):
        raise ValueError(f"Allegro v5 action must be finite shape (16,), got {action.shape}")
    data = _load_allegro_v5_direct_anchor_data()
    calibrated = np.empty_like(action)
    has_anchor_target = False
    tip_feature = None
    if manus_tip_feature is not None:
        candidate = np.asarray(manus_tip_feature, dtype=np.float64)
        if candidate.shape == (15,) and np.all(np.isfinite(candidate)):
            tip_feature = candidate
    ergonomic_feature = None
    if manus_ergonomic_feature is not None:
        candidate = np.asarray(manus_ergonomic_feature, dtype=np.float64)
        if candidate.shape == (6,) and np.all(np.isfinite(candidate)):
            ergonomic_feature = candidate

    # An exactly replayed capture must reconstruct its manually edited target
    # as one complete pose.  Finger-local features can be intentionally
    # identical across different captures, so without this exact-anchor rule
    # their conflicting manual targets cannot all be reproduced.
    if tip_feature is not None:
        exact_anchor = np.all(np.isclose(data["raw_features"], action, atol=1e-12, rtol=0.0), axis=1)
        exact_anchor &= np.all(
            np.isclose(data["manus_tip_features"], tip_feature, atol=1e-12, rtol=0.0),
            axis=1,
        )
        if ergonomic_feature is not None:
            exact_anchor &= np.all(
                np.isclose(
                    data["manus_ergonomic_features"], ergonomic_feature,
                    atol=1e-12,
                    rtol=0.0,
                ),
                axis=1,
            )
        matching_indices = np.flatnonzero(exact_anchor)
        if matching_indices.size:
            return clip_allegro_v5_safe_action(data["targets"][matching_indices[0]])

    if tip_feature is not None:
        manus_tip_feature = tip_feature
        for _finger, action_slice, tip_slice, _ergonomic_slice in (
            _ALLEGRO_V5_FINGER_INTERPOLATION_LAYOUT
        ):
            anchor_feature = np.concatenate(
                (
                    data["manus_tip_features"][:, tip_slice],
                    data["raw_features"][:, action_slice]
                    * _ALLEGRO_V5_TIP_RAW_TIE_BREAK_SCALE_M_PER_RAD,
                ),
                axis=1,
            )
            query_feature = np.concatenate(
                (
                    manus_tip_feature[tip_slice],
                    action[action_slice]
                    * _ALLEGRO_V5_TIP_RAW_TIE_BREAK_SCALE_M_PER_RAD,
                )
            )
            tip_target, _tip_influence = _distance_gated_target(
                anchor_feature,
                query_feature,
                data["targets"][:, action_slice],
                sigma=data["sigma"],
                influence_radius=data["influence_radius"],
                epsilon=data["inverse_distance_epsilon"],
                distance_power=data["inverse_distance_power"],
            )
            calibrated[action_slice] = tip_target
            has_anchor_target = True

    if ergonomic_feature is not None:
        manus_ergonomic_feature = ergonomic_feature
        for _finger, action_slice, _tip_slice, ergonomic_slice in (
            _ALLEGRO_V5_FINGER_INTERPOLATION_LAYOUT
        ):
            # Ergonomic degrees provide the metric; the local raw action
            # merely disambiguates otherwise identical MANUS angles.
            anchor_feature = np.concatenate(
                (
                    data["manus_ergonomic_features"][:, ergonomic_slice],
                    data["raw_features"][:, action_slice],
                ),
                axis=1,
            )
            query_feature = np.concatenate(
                (
                    manus_ergonomic_feature[ergonomic_slice], action[action_slice]
                )
            )
            ergonomic_target, ergonomic_influence = _distance_gated_target(
                anchor_feature,
                query_feature,
                data["targets"][:, action_slice],
                sigma=data["ergonomic_sigma"],
                influence_radius=data["ergonomic_influence_radius"],
                epsilon=data["ergonomic_inverse_distance_epsilon"],
                distance_power=data["ergonomic_inverse_distance_power"],
            )
            if tip_feature is None:
                calibrated[action_slice] = ergonomic_target
            else:
                # This is still a blend of the same seven target poses: named
                # MANUS angles only refine the finger-local anchor weighting.
                influence = (
                    _ALLEGRO_V5_ERGONOMIC_COMPATIBILITY_BLEND * ergonomic_influence
                )
                calibrated[action_slice] = calibrated[action_slice] + influence * (
                    ergonomic_target - calibrated[action_slice]
                )
            has_anchor_target = True
    if not has_anchor_target:
        # Do not resurrect the old direct-angle output when MANUS data is
        # malformed.  Select the closest of the seven approved targets instead.
        nearest_anchor = np.argmin(
            np.sum((data["raw_features"] - action[None, :]) ** 2, axis=1)
        )
        calibrated = data["targets"][nearest_anchor].copy()
    return clip_allegro_v5_safe_action(calibrated)


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


def _load_wuji_direct_anchor_data(is_right=True):
    """Load operator-edited Wuji anchors for the captured hand side.

    Only the left-hand calibration exists currently.  The uncalibrated side
    intentionally keeps the established direct retargeter unchanged.
    """
    side = "right" if is_right else "left"
    if side == "right":
        return None
    if side in _WUJI_DIRECT_ANCHOR_DATA:
        return _WUJI_DIRECT_ANCHOR_DATA[side]

    with _WUJI_DIRECT_ANCHOR_CONFIG.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if (
        raw.get("schema_version") != 1
        or raw.get("method")
        != "wuji_direct_manus_palm_normalized_tip_shepard_residual"
        or raw.get("side") != side
    ):
        raise RuntimeError("Unsupported Wuji direct anchor calibration")

    interpolation = raw.get("interpolation", {})
    data = {
        "raw_features": np.asarray(raw["anchor_raw_actions_rad"], dtype=np.float64),
        "manus_tip_features": np.asarray(
            raw["anchor_manus_tip_features_m"], dtype=np.float64
        ),
        "targets": np.asarray(raw["anchor_target_actions_rad"], dtype=np.float64),
        "safe_lower": np.asarray(raw["safe_lower_rad"], dtype=np.float64),
        "safe_upper": np.asarray(raw["safe_upper_rad"], dtype=np.float64),
        "sigma": float(interpolation["gaussian_sigma_palm_lengths"]),
        "influence_radius": float(interpolation["influence_radius_palm_lengths"]),
        "inverse_distance_epsilon": float(
            interpolation["inverse_distance_epsilon_palm_lengths_sq"]
        ),
        "inverse_distance_power": float(interpolation["inverse_distance_power"]),
        "duplicate_cluster_radius": float(
            interpolation["duplicate_cluster_radius_palm_lengths"]
        ),
    }
    lower, upper = _resolve_wuji_direct_limits(is_right=is_right)
    if (
        data["raw_features"].ndim != 2
        or data["raw_features"].shape[0] < 2
        or data["raw_features"].shape[1] != 20
        or data["manus_tip_features"].shape != (data["raw_features"].shape[0], 15)
        or data["targets"].shape != (data["raw_features"].shape[0], 20)
        or data["safe_lower"].shape != (20,)
        or data["safe_upper"].shape != (20,)
        or not all(np.all(np.isfinite(data[key])) for key in (
            "raw_features", "manus_tip_features", "targets", "safe_lower", "safe_upper"
        ))
        or not np.all(data["safe_lower"] <= data["safe_upper"])
        or not np.all(lower <= data["safe_lower"])
        or not np.all(data["safe_upper"] <= upper)
        or not all(np.isfinite(data[key]) and data[key] > 0.0 for key in (
            "sigma", "influence_radius", "inverse_distance_epsilon"
        ))
        or not np.isfinite(data["inverse_distance_power"])
        or data["inverse_distance_power"] < 1.0
        or not np.isfinite(data["duplicate_cluster_radius"])
        or data["duplicate_cluster_radius"] <= 0.0
    ):
        raise RuntimeError("Wuji direct anchor calibration is invalid")

    _WUJI_DIRECT_ANCHOR_DATA[side] = data
    return data


def clip_wuji_direct_safe_action(action, is_right=True):
    """Keep Wuji direct commands within its URDF physical safety limits."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (20,):
        raise ValueError(f"Wuji action must have shape (20,), got {action.shape}")
    data = _load_wuji_direct_anchor_data(is_right=is_right)
    if data is None:
        lower, upper = _resolve_wuji_direct_limits(is_right=is_right)
    else:
        lower, upper = data["safe_lower"], data["safe_upper"]
    return np.clip(action, lower, upper).astype(np.float32)


def _wuji_direct_tip_feature(keypoints):
    """Return palm-normalized wrist-relative MANUS fingertip coordinates.

    ``wuji_direct`` derives joint angles from directions, so those raw angles
    are invariant to the caller's optional hand scale.  Normalize the tip
    anchor feature by the wrist-to-middle-MCP distance for the same property:
    captures made in the alignment UI at ``--scale 1.15`` must also replay
    through the standard Retargetor default scale of ``1.0``.
    """
    keypoints = np.asarray(keypoints, dtype=np.float64)
    if keypoints.shape != (21, 3) or not np.all(np.isfinite(keypoints)):
        return None
    palm_scale = np.linalg.norm(keypoints[9] - keypoints[0])
    if not np.isfinite(palm_scale) or palm_scale < 1e-6:
        return None
    return ((keypoints[[4, 8, 12, 16, 20]] - keypoints[0]) / palm_scale).reshape(15)


def _align_wuji_direct_action(action, keypoints, is_right=True):
    """Locally blend edited Wuji targets over the established direct output."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (20,) or not np.all(np.isfinite(action)):
        raise ValueError(f"Wuji action must be finite shape (20,), got {action.shape}")
    data = _load_wuji_direct_anchor_data(is_right=is_right)
    tip_feature = _wuji_direct_tip_feature(keypoints)
    if data is None or tip_feature is None:
        return clip_wuji_direct_safe_action(action, is_right=is_right)

    # Preserve every manually edited capture exactly.  Local per-finger
    # interpolation alone cannot resolve two captures whose fingertip feature
    # happens to coincide but whose edited robot targets differ.
    # Direction-derived angles are scale-invariant in theory, but a nearly
    # straight distal segment can differ by about 3.5e-4 rad after float32
    # arithmetic when the caller changes ``hand_scale``.  The normalized tip
    # feature remains the strict pose identity check; this tolerance only
    # prevents that numerical detail from defeating exact capture replay.
    exact_anchor = np.all(
        np.isclose(data["raw_features"], action, atol=5e-4, rtol=0.0), axis=1
    ) & np.all(
        np.isclose(data["manus_tip_features"], tip_feature, atol=2e-6, rtol=0.0),
        axis=1,
    )
    matching_indices = np.flatnonzero(exact_anchor)
    if matching_indices.size:
        return clip_wuji_direct_safe_action(
            data["targets"][matching_indices[0]], is_right=is_right
        )

    calibrated = action.copy()
    for action_slice, tip_slice in _WUJI_DIRECT_FINGER_INTERPOLATION_LAYOUT:
        anchor_feature = np.concatenate(
            (
                data["manus_tip_features"][:, tip_slice],
                data["raw_features"][:, action_slice]
                * _WUJI_DIRECT_TIP_RAW_TIE_BREAK_SCALE_PALM_LENGTHS_PER_RAD,
            ),
            axis=1,
        )
        query_feature = np.concatenate(
            (
                tip_feature[tip_slice],
                action[action_slice]
                * _WUJI_DIRECT_TIP_RAW_TIE_BREAK_SCALE_PALM_LENGTHS_PER_RAD,
            )
        )
        target, influence = _cluster_balanced_distance_gated_target(
            anchor_feature,
            query_feature,
            data["targets"][:, action_slice],
            sigma=data["sigma"],
            influence_radius=data["influence_radius"],
            epsilon=data["inverse_distance_epsilon"],
            distance_power=data["inverse_distance_power"],
            duplicate_cluster_radius=data["duplicate_cluster_radius"],
        )
        calibrated[action_slice] = action[action_slice] + influence * (
            target - action[action_slice]
        )
    return clip_wuji_direct_safe_action(calibrated, is_right=is_right)


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
    raw_action = np.clip(angles.reshape(20), lower, upper).astype(np.float32)
    return _align_wuji_direct_action(raw_action, keypoints, is_right=is_right)


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

_INSPIRE_FINGER_ANGLE_RANGE_DEG = 176.7 - 19.0
_INSPIRE_THUMB_BEND_RANGE_DEG = 53.6 - (-13.0)
_INSPIRE_THUMB_ROTATE_RANGE_DEG = 165.0 - 90.0
_INSPIRE_THUMB_EXTRA_BEND_GAIN = 3.0
_INSPIRE_THUMB_EXTRA_BEND_COMMAND = -500.0
_INSPIRE_THUMB_SECONDARY_GAIN = 0.8


def _open_to_closed_command(flexion_deg, range_deg):
    normalized = np.clip(float(flexion_deg) / float(range_deg), 0.0, 1.0)
    return (1.0 - normalized) * 1000.0


def _inspire_thumb_secondary_fraction(flexion_deg):
    """Return Inspire's secondary-thumb command as a [0, 0.8] fraction."""
    return (
        1000.0
        - _open_to_closed_command(flexion_deg, _INSPIRE_THUMB_BEND_RANGE_DEG)
    ) * (_INSPIRE_THUMB_SECONDARY_GAIN / 1000.0)


def inspire_from_manus_ergonomics(ergonomics):
    """Map MANUS ergonomic angles to RH56 ANGLE_SET register order."""
    required = []
    for finger in ("Pinky", "Ring", "Middle", "Index"):
        required.extend(
            (
                f"{finger}MCPStretch",
                f"{finger}PIPStretch",
                f"{finger}DIPStretch",
            )
        )
    required.extend(("ThumbMCPStretch", "ThumbMCPSpread"))
    missing = [name for name in required if name not in ergonomics]
    if missing:
        raise ValueError(
            "MANUS ergonomics is missing Inspire inputs: "
            + ", ".join(missing)
        )

    command = np.zeros(6, dtype=np.float64)
    for index, finger in enumerate(("Pinky", "Ring", "Middle", "Index")):
        flexion = sum(
            float(ergonomics[f"{finger}{joint}Stretch"])
            for joint in ("MCP", "PIP", "DIP")
        )
        command[index] = _open_to_closed_command(
            flexion,
            _INSPIRE_FINGER_ANGLE_RANGE_DEG,
        )

    command[4] = np.clip(
        (
            np.clip(
                float(ergonomics["ThumbMCPSpread"])
                / _INSPIRE_THUMB_ROTATE_RANGE_DEG,
                0.0,
                1.0,
            )
            * 1000.0
            * _INSPIRE_THUMB_EXTRA_BEND_GAIN
        )
        + _INSPIRE_THUMB_EXTRA_BEND_COMMAND,
        0.0,
        1000.0,
    )
    command[5] = np.clip(
        _inspire_thumb_secondary_fraction(ergonomics["ThumbMCPStretch"])
        * 1000.0,
        0.0,
        1000.0,
    )
    return command


def inspire(
    hand_pose_frame,
    is_right=True,
    ergonomics=None,
):
    if ergonomics is not None:
        return inspire_from_manus_ergonomics(ergonomics)

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
            thumb_forward = (
                tip_direction[0] > 0
                if is_right
                else tip_direction[0] < 0
            )
            if thumb_forward:
                inspire_angles[5] = 1000 - np.arctan(-tip_direction[2] / abs(tip_direction[0])) / np.pi * 2000
                inspire_angles[4] = np.arccos(-tip_direction[1]) * 2000 - 1000 # no divide by pi for better range
            else:
                inspire_angles[5] = 0
                thumb_scale = 3.8 if is_right else 3.5
                inspire_angles[4] = np.arcsin(-tip_direction[2]) / np.pi * 2000 * thumb_scale - 1000
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

    closed_distance = 0.04
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


def _allegro_v5_raw_from_manus_ergonomics(ergonomics):
    """Convert named MANUS joint angles to the legacy raw-v5 action order.

    VIVE's MANUS stream contains stable per-joint ergonomics in degrees.  The
    former raw extractor inferred these quantities from reconstructed 4x4
    transforms, which is both noisier and sensitive to the VIVE wrist frame.
    This is the direct equivalent of :func:`inspire_from_manus_ergonomics`:
    it uses the MANUS angles whenever the complete named payload is present.

    The non-thumb MCP bend keeps the legacy ``(angle - 0.35) * 1.5``
    calibration, while PIP/DIP retain their direct radians.  The thumb spread
    and distal bends preserve the signs/scales of the old Euler/rotation-matrix
    extraction.  Returning the existing index, middle, ring, thumb order is
    essential because the v5 ROS controller consumes that order directly.
    """
    if not isinstance(ergonomics, dict):
        raise ValueError("MANUS ergonomics must be a name-to-angle mapping")
    missing = [
        field for field in _ALLEGRO_V5_RAW_ERGONOMIC_FIELDS if field not in ergonomics
    ]
    if missing:
        raise ValueError(
            "MANUS ergonomics is missing Allegro v5 raw inputs: "
            + ", ".join(missing)
        )
    try:
        values = {
            field: float(ergonomics[field])
            for field in _ALLEGRO_V5_RAW_ERGONOMIC_FIELDS
        }
    except (TypeError, ValueError) as error:
        raise ValueError("MANUS ergonomics contains a non-numeric Allegro input") from error
    if not np.all(np.isfinite(tuple(values.values()))):
        raise ValueError("MANUS ergonomics contains a non-finite Allegro input")

    action = np.zeros(16, dtype=np.float64)
    for index, finger in enumerate(("Index", "Middle", "Ring")):
        offset = index * 4
        # MANUS calls the first joint's lateral component ``Spread``.  It is
        # already a signed angular quantity; preserve it in radians rather
        # than deriving it from fingertip position in the VIVE wrist frame.
        action[offset] = np.deg2rad(values[f"{finger}Spread"])
        action[offset + 1] = (
            np.deg2rad(values[f"{finger}MCPStretch"]) - 0.35
        ) * 1.5
        action[offset + 2] = np.deg2rad(values[f"{finger}PIPStretch"])
        action[offset + 3] = np.deg2rad(values[f"{finger}DIPStretch"])

    # Reuse Inspire's ThumbMCPStretch normalization and 0.8 gain, but reverse
    # its direction for Allegro joint 12: increasing MANUS stretch decreases
    # this Allegro joint.  Convert the resulting fraction to radians only
    # after applying that direction change.
    action[12] = (
        _INSPIRE_THUMB_SECONDARY_GAIN
        - _inspire_thumb_secondary_fraction(values["ThumbMCPStretch"])
    ) * _ALLEGRO_V5_PHYSICAL_UPPER[12]
    action[13] = (
        -np.deg2rad(values["ThumbMCPSpread"])
        * _ALLEGRO_V5_THUMB_SECONDARY_FLEX_GAIN
        - 1.57
    )
    action[14] = (
        np.sin(np.deg2rad(values["ThumbPIPStretch"]))
        * 1.2
        * _ALLEGRO_V5_THUMB_SECONDARY_FLEX_GAIN
    )
    action[15] = (
        np.sin(np.deg2rad(values["ThumbDIPStretch"]))
        * 1.2
        * _ALLEGRO_V5_THUMB_SECONDARY_FLEX_GAIN
    )
    return action


def _allegro_v5_raw(hand_pose_frame, ergonomics=None):
    """Extract raw Allegro-v5 angles in retargeter action order.

    Prefer named MANUS ergonomics for VIVE teleoperation.  The transform
    implementation remains only as the compatibility fallback for older
    MANUS sources that do not publish ergonomics.
    """
    if ergonomics is not None:
        return _allegro_v5_raw_from_manus_ergonomics(ergonomics)
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
            # ``arctan2`` avoids a discontinuity or divide-by-zero warning
            # when a fingertip direction crosses the local z=0 plane.
            allegro_angles[4*i] = np.arctan2(
                tip_direction[0], tip_direction[2]
            ) * (0.9 - tip_direction[1])
        
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


def allegro_v5(hand_pose_frame, ergonomics=None):
    """Retarget MANUS data through the raw-v5 action contract.

    VIVE teleoperation supplies named ergonomic joint angles, so the raw
    mapper uses those directly.  Older transform-only MANUS inputs still use
    the fallback inside :func:`_allegro_v5_raw`.
    """
    return _allegro_v5_raw(hand_pose_frame, ergonomics=ergonomics)
