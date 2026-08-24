import json
import os
import time
from functools import partial
from threading import Event, Lock, Thread

import numpy as np


_FINGER_NAMES = {"thumb", "index", "middle", "ring", "pinky"}
_THUMB_JOINT_NAMES = {
    "mcp": "metacarpal",
    "pip": "proximal",
    "dip": "distal",
    "tip": "tip",
}
_FINGER_JOINT_NAMES = {
    "mcp": "metacarpal",
    "pip": "proximal",
    "ip": "intermediate",
    "dip": "distal",
    "tip": "tip",
}
_FINGER_CHAINS = {
    "thumb": (
        "thumb_metacarpal",
        "thumb_proximal",
        "thumb_distal",
        "thumb_tip",
    ),
    **{
        finger: (
            f"{finger}_metacarpal",
            f"{finger}_proximal",
            f"{finger}_intermediate",
            f"{finger}_distal",
            f"{finger}_tip",
        )
        for finger in ("index", "middle", "ring", "pinky")
    },
}
_VIVE_TRACKER_TO_WRIST_ROTATION = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
_VIVE_TRANSLATION_AXIS_TRANSFORM = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def pose_to_matrix(pose, allow_invalid_orientation=False):
    """Convert a geometry_msgs Pose-like object to a finite 4x4 transform."""
    if hasattr(pose, "pose"):
        pose = pose.pose

    position = np.array(
        [pose.position.x, pose.position.y, pose.position.z],
        dtype=np.float64,
    )
    quaternion = np.array(
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(position)):
        raise ValueError("Pose position contains non-finite values.")

    norm = np.linalg.norm(quaternion)
    transform = np.eye(4, dtype=np.float64)
    if np.all(np.isfinite(quaternion)) and norm >= 1.0e-9:
        x, y, z, w = quaternion / norm
        transform[:3, :3] = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
    elif not allow_invalid_orientation:
        raise ValueError("Pose quaternion is non-finite or has zero norm.")
    transform[:3, 3] = position
    return transform


def apply_vive_tracker_mount_rotation(transform):
    """Apply the configured VIVE rotation and translation-axis corrections."""
    corrected = transform.copy()
    corrected[:3, :3] = (
        corrected[:3, :3] @ _VIVE_TRACKER_TO_WRIST_ROTATION
    )
    corrected[:3, 3] = (
        _VIVE_TRANSLATION_AXIS_TRANSFORM @ corrected[:3, 3]
    )
    return corrected


def _canonical_joint_name(chain_type, joint_type):
    finger = str(chain_type).strip().lower()
    joint = str(joint_type).strip().lower()
    if finger not in _FINGER_NAMES:
        return None

    mapping = _THUMB_JOINT_NAMES if finger == "thumb" else _FINGER_JOINT_NAMES
    suffix = mapping.get(joint)
    return f"{finger}_{suffix}" if suffix is not None else None


def _unit_vector(vector, fallback):
    norm = np.linalg.norm(vector)
    if np.isfinite(norm) and norm >= 1.0e-9:
        return vector / norm
    return np.asarray(fallback, dtype=np.float64)


def _frame_from_bone_direction(direction, palm_normal):
    y_axis = _unit_vector(direction, (0.0, 1.0, 0.0))
    x_axis = _unit_vector(
        np.cross(y_axis, palm_normal),
        (1.0, 0.0, 0.0),
    )
    z_axis = _unit_vector(
        np.cross(x_axis, y_axis),
        (0.0, 0.0, 1.0),
    )
    x_axis = _unit_vector(np.cross(y_axis, z_axis), x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def _reconstruct_hand_orientations(frame):
    wrist_position = frame["wrist"][:3, 3]
    across = (
        frame["index_metacarpal"][:3, 3]
        - frame["pinky_metacarpal"][:3, 3]
    )
    forward = (
        frame["middle_metacarpal"][:3, 3]
        - wrist_position
    )
    wrist_y = _unit_vector(forward, (0.0, 1.0, 0.0))
    wrist_x = across - np.dot(across, wrist_y) * wrist_y
    wrist_x = _unit_vector(wrist_x, (1.0, 0.0, 0.0))
    wrist_z = _unit_vector(
        np.cross(wrist_x, wrist_y),
        (0.0, 0.0, 1.0),
    )
    wrist_x = _unit_vector(np.cross(wrist_y, wrist_z), wrist_x)
    wrist_rotation = np.column_stack((wrist_x, wrist_y, wrist_z))
    frame["wrist"][:3, :3] = wrist_rotation

    for names in _FINGER_CHAINS.values():
        available = [name for name in names if name in frame]
        previous_direction = wrist_y
        for index, name in enumerate(available):
            position = frame[name][:3, 3]
            if index + 1 < len(available):
                direction = frame[available[index + 1]][:3, 3] - position
                previous_direction = _unit_vector(direction, previous_direction)
            else:
                direction = previous_direction
            frame[name][:3, :3] = _frame_from_bone_direction(
                direction,
                wrist_z,
            )


def manus_message_to_frame(message):
    """Convert a ManusGlove message into the named pose contract used by Xsens."""
    frame = {}
    wrist = None

    for node in message.raw_nodes:
        transform = pose_to_matrix(node.pose, allow_invalid_orientation=True)
        chain = str(node.chain_type).strip().lower()
        if chain == "hand" or int(node.node_id) == 0:
            wrist = transform
            continue

        name = _canonical_joint_name(node.chain_type, node.joint_type)
        if name is not None:
            frame[name] = transform

    if wrist is None:
        raise ValueError("MANUS frame does not contain a wrist/hand node.")

    required = {
        "thumb_metacarpal",
        "thumb_proximal",
        "thumb_distal",
    }
    for finger in ("index", "middle", "ring", "pinky"):
        required.update(
            {
                f"{finger}_metacarpal",
                f"{finger}_proximal",
                f"{finger}_intermediate",
                f"{finger}_distal",
            }
        )
    missing = sorted(required.difference(frame))
    if missing:
        raise ValueError(f"MANUS frame is missing joints: {', '.join(missing)}")

    frame["wrist"] = wrist
    _reconstruct_hand_orientations(frame)
    return frame


def manus_ergonomics_to_dict(ergonomics):
    """Copy finite MANUS ergonomics angles into a name-to-degrees mapping."""
    values = {}
    for item in ergonomics:
        name = str(item.type)
        value = float(item.value)
        if np.isfinite(value):
            values[name] = value
    return values


def reparent_frame(frame, wrist):
    """Place MANUS wrist-relative finger transforms under a VIVE wrist."""
    source_wrist_inv = np.linalg.inv(frame["wrist"])
    result = {"wrist": wrist.copy()}
    for name, transform in frame.items():
        if name != "wrist":
            result[name] = wrist @ source_wrist_inv @ transform
    return result


def _copy_frame(frame):
    return {name: transform.copy() for name, transform in frame.items()}


def _pause_state_from_ergonomics(ergonomics, straight_threshold=40.0):
    values = {str(item.type): float(item.value) for item in ergonomics}
    scores = []
    for finger in ("Index", "Middle", "Ring", "Pinky"):
        required = (
            f"{finger}MCPStretch",
            f"{finger}PIPStretch",
        )
        if any(name not in values or not np.isfinite(values[name]) for name in required):
            return None
        scores.append(sum(values[name] for name in required) / len(required))
    return 0 if all(score >= straight_threshold for score in scores) else 1


class ViveManusROSReceiver:
    """Receive MANUS hands, optionally fusing a VIVE wrist tracker."""

    def __init__(
        self,
        vive_right_topic="/vive_trackers/right_hand/pose",
        manus_topics=("/manus_glove_0", "/manus_glove_1"),
        max_age_s=0.25,
        hand_side="right",
        vive_left_topic="/vive_trackers/left_hand/pose",
        require_left_control=True,
        use_vive=True,
        use_manus=True,
    ):
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
        except ImportError as exc:
            raise RuntimeError(
                "VIVE/MANUS teleop requires ROS 2 Humble and the vive-teleop workspace. "
                "Source /home/temp_id/vive-teleop/ros2_ws/install/setup.bash first."
            ) from exc
        if use_manus:
            try:
                from manus_ros2_msgs.msg import ManusGlove
            except ImportError as exc:
                raise RuntimeError(
                    "MANUS teleop requires manus_ros2_msgs.  For VIVE-only "
                    "arm teleoperation, pass use_manus=False."
                ) from exc
        if use_vive:
            try:
                from geometry_msgs.msg import PoseStamped
            except ImportError as exc:
                raise RuntimeError("VIVE wrist tracking requires geometry_msgs.") from exc

        self.hand_side = str(hand_side).strip().lower()
        valid_hand_sides = ("right", "bimanual") if use_vive else ("right", "left", "bimanual")
        if self.hand_side not in valid_hand_sides:
            raise ValueError(
                "VIVE teleop supports hand_side='right' or 'bimanual'; "
                "MANUS-only mode also supports hand_side='left'."
            )

        self.vive_right_topic = str(vive_right_topic)
        self.vive_left_topic = str(vive_left_topic)
        self.manus_topics = tuple(str(topic) for topic in manus_topics)
        # Legacy capture sessions use the left glove as a pause/control input
        # even when actuating only the right hand.  Alignment tools do not.
        self.require_left_control = bool(require_left_control)
        self.use_vive = bool(use_vive)
        self.use_manus = bool(use_manus)
        if not self.use_vive and not self.use_manus:
            raise ValueError("At least one of VIVE or MANUS input must be enabled.")
        if not self.use_manus and self.require_left_control:
            raise ValueError("VIVE-only teleoperation cannot require MANUS left-hand control.")
        self.max_age_s = float(max_age_s)
        if not np.isfinite(self.max_age_s) or self.max_age_s <= 0:
            raise ValueError("max_age_s must be a positive finite value.")

        self._rclpy = rclpy
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()

        self.lock = Lock()
        self.save_event = Event()
        self.error_event = Event()
        self.exit_event = Event()
        self._vive_right = None
        self._vive_right_time = None
        self._vive_frame_id = None
        self._vive_left = None
        self._vive_left_time = None
        self._vive_left_frame_id = None
        self._manus_frames = {"Left": None, "Right": None}
        self._manus_ergonomics = {"Left": None, "Right": None}
        self._manus_times = {"Left": None, "Right": None}
        self._manus_glove_ids = {"Left": None, "Right": None}
        # Keep transport diagnostics separately from valid frames.  A MANUS
        # publisher can be alive while its messages are malformed or report a
        # different hand side; callers need to be able to distinguish those
        # cases from a topic that has no traffic at all.
        self._manus_topic_message_counts = {
            topic: 0 for topic in self.manus_topics
        }
        self._manus_valid_message_counts = {"Left": 0, "Right": 0}
        self._manus_invalid_message_counts = {"Left": 0, "Right": 0}
        self._manus_last_topics = {"Left": None, "Right": None}
        self._manus_last_errors = {"Left": None, "Right": None}
        self._manus_unknown_side_count = 0
        self._manus_last_unknown_side = None
        self._pause_state = None
        self.save_path = None
        self.data = None

        node_name = f"vive_manus_teleop_receiver_{os.getpid()}"
        self.node = rclpy.create_node(node_name)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.subscriptions = []
        if self.use_vive:
            self.subscriptions.append(
                self.node.create_subscription(
                    PoseStamped,
                    self.vive_right_topic,
                    partial(self._vive_callback, side="Right"),
                    qos_profile_sensor_data,
                )
            )
        if self.use_vive and self.hand_side == "bimanual":
            self.subscriptions.append(
                self.node.create_subscription(
                    PoseStamped,
                    self.vive_left_topic,
                    partial(self._vive_callback, side="Left"),
                    qos_profile_sensor_data,
                )
            )
        if self.use_manus:
            for topic in self.manus_topics:
                self.subscriptions.append(
                    self.node.create_subscription(
                        ManusGlove,
                        topic,
                        lambda message, source_topic=topic: self._manus_callback(
                            message, source_topic
                        ),
                        qos_profile_sensor_data,
                    )
                )

        self.spin_thread = Thread(target=self._spin, daemon=True)
        self.spin_thread.start()
        wrist_topics = (
            f"right wrist={self.vive_right_topic}"
            if self.use_vive
            else "MANUS wrist frame (no VIVE)"
        )
        if self.use_vive and self.hand_side == "bimanual":
            wrist_topics = (
                f"left wrist={self.vive_left_topic}, {wrist_topics}"
            )
        input_description = (
            f"gloves={', '.join(self.manus_topics)}"
            if self.use_manus
            else "MANUS disabled"
        )
        self.node.get_logger().info(
            f"VIVE/MANUS teleop ({self.hand_side}): "
            f"{wrist_topics}, {input_description}"
        )

    def _spin(self):
        try:
            self.executor.spin()
        except Exception as exc:
            if not self.exit_event.is_set():
                self.node.get_logger().error(f"ROS 2 spin failed: {exc}")
                self.error_event.set()

    def _vive_callback(self, message, side):
        try:
            transform = apply_vive_tracker_mount_rotation(
                pose_to_matrix(message.pose)
            )
        except (AttributeError, TypeError, ValueError) as exc:
            self.node.get_logger().warning(
                f"Ignoring invalid {side} VIVE pose: {exc}"
            )
            return

        with self.lock:
            timestamp = time.monotonic()
            frame_id = str(message.header.frame_id)
            if side == "Right":
                self._vive_right = transform
                self._vive_right_time = timestamp
                self._vive_frame_id = frame_id
            elif side == "Left":
                self._vive_left = transform
                self._vive_left_time = timestamp
                self._vive_left_frame_id = frame_id
            else:
                raise ValueError(f"Unsupported VIVE tracker side: {side}")

    def _manus_callback(self, message, source_topic):
        timestamp = time.monotonic()
        with self.lock:
            self._manus_topic_message_counts[source_topic] = (
                self._manus_topic_message_counts.get(source_topic, 0) + 1
            )

        side = str(getattr(message, "side", "")).strip().lower().capitalize()
        if side not in self._manus_frames:
            with self.lock:
                self._manus_unknown_side_count += 1
                self._manus_last_unknown_side = str(getattr(message, "side", ""))
            self.node.get_logger().warning(
                f"Ignoring MANUS message with invalid side '{getattr(message, 'side', '')}' "
                f"from {source_topic}"
            )
            return

        try:
            frame = manus_message_to_frame(message)
        except (AttributeError, TypeError, ValueError) as exc:
            with self.lock:
                self._manus_invalid_message_counts[side] += 1
                self._manus_last_errors[side] = str(exc)
            self.node.get_logger().warning(
                f"Ignoring invalid {side} MANUS frame from {source_topic}: {exc}"
            )
            return

        with self.lock:
            self._manus_frames[side] = frame
            self._manus_ergonomics[side] = manus_ergonomics_to_dict(
                message.ergonomics
            )
            self._manus_times[side] = timestamp
            self._manus_glove_ids[side] = int(message.glove_id)
            self._manus_valid_message_counts[side] += 1
            self._manus_last_topics[side] = source_topic
            self._manus_last_errors[side] = None
            if side == "Left":
                self._pause_state = _pause_state_from_ergonomics(
                    message.ergonomics
                )

    def get_diagnostics(self):
        """Return non-blocking MANUS transport state for operator-facing UIs."""
        now_monotonic = time.monotonic()
        with self.lock:
            def _side_diagnostics(side):
                timestamp = self._manus_times[side]
                age_s = (
                    None if timestamp is None
                    else max(0.0, now_monotonic - timestamp)
                )
                return {
                    "valid_messages": self._manus_valid_message_counts[side],
                    "invalid_messages": self._manus_invalid_message_counts[side],
                    "last_age_s": age_s,
                    "is_fresh": age_s is not None and age_s <= self.max_age_s,
                    "last_topic": self._manus_last_topics[side],
                    "glove_id": self._manus_glove_ids[side],
                    "last_error": self._manus_last_errors[side],
                }

            return {
                "manus_topics": list(self.manus_topics),
                "topic_message_counts": dict(self._manus_topic_message_counts),
                "unknown_side_messages": self._manus_unknown_side_count,
                "last_unknown_side": self._manus_last_unknown_side,
                "Right": _side_diagnostics("Right"),
                "Left": _side_diagnostics("Left"),
            }

    def get_data(self):
        now_monotonic = time.monotonic()
        now_wall = time.time()
        # ``getattr`` keeps the long-standing lightweight test/dummy receiver
        # contract working: before MANUS-only mode existed, an object with
        # only the legacy fields necessarily meant VIVE mode with left control.
        use_vive = getattr(self, "use_vive", True)
        use_manus = getattr(self, "use_manus", True)
        require_left_control = getattr(self, "require_left_control", True)
        if not use_manus:
            if not use_vive:
                return {"Left": None, "Right": None, "time": now_wall}
            with self.lock:
                right_fresh = (
                    self._vive_right_time is not None
                    and now_monotonic - self._vive_right_time <= self.max_age_s
                )
                left_fresh = (
                    self._vive_left_time is not None
                    and now_monotonic - self._vive_left_time <= self.max_age_s
                )
                if not right_fresh or (
                    self.hand_side == "bimanual" and not left_fresh
                ):
                    return {"Left": None, "Right": None, "time": now_wall}
                right_frame = {"wrist": self._vive_right.copy()}
                left_frame = (
                    {"wrist": self._vive_left.copy()}
                    if self.hand_side == "bimanual"
                    else None
                )
                source_times = {
                    "vive_right": self._vive_right_time,
                }
                if self.hand_side == "bimanual":
                    source_times["vive_left"] = self._vive_left_time

            if self.save_event.is_set():
                with self.lock:
                    if self.data is not None:
                        self.data["time"].append(now_wall)
                        self.data["Left"].append(
                            None if left_frame is None else _copy_frame(left_frame)
                        )
                        self.data["Right"].append(_copy_frame(right_frame))
                        self.data["vive_right_time"].append(source_times["vive_right"])
                        if self.hand_side == "bimanual":
                            self.data["vive_left_time"].append(
                                source_times["vive_left"]
                            )
            return {
                "Left": left_frame,
                "Right": right_frame,
                "ergonomics": {},
                "time": now_wall,
            }
        # A left-only stream is meaningful only without VIVE wrist fusion:
        # there is no separate unimanual left VIVE contract.  Do not require
        # a right glove merely because legacy right-hand teleop did.
        left_only_manus = self.hand_side == "left" and not use_vive
        if left_only_manus:
            with self.lock:
                timestamp = self._manus_times["Left"]
                fresh = (
                    timestamp is not None
                    and now_monotonic - timestamp <= self.max_age_s
                )
                if not fresh:
                    return {"Left": None, "Right": None, "time": now_wall}
                left_frame = _copy_frame(self._manus_frames["Left"])
                left_ergonomics = self._manus_ergonomics["Left"].copy()
            return {
                "Left": left_frame,
                "Right": None,
                "ergonomics": {"Left": left_ergonomics},
                "time": now_wall,
            }
        with self.lock:
            timestamps = [
                self._manus_times["Right"],
            ]
            if use_vive:
                timestamps.append(self._vive_right_time)
            if self.hand_side == "bimanual" or require_left_control:
                timestamps.extend(
                    [self._manus_times["Left"]]
                )
            if use_vive and self.hand_side == "bimanual":
                timestamps.append(self._vive_left_time)
            ready = all(timestamp is not None for timestamp in timestamps)
            fresh = ready and all(
                now_monotonic - timestamp <= self.max_age_s
                for timestamp in timestamps
            )
            if not fresh:
                return {"Left": None, "Right": None, "time": now_wall}

            vive_right = self._vive_right.copy() if use_vive else None
            vive_left = (
                self._vive_left.copy()
                if use_vive and self.hand_side == "bimanual"
                else None
            )
            right_manus = _copy_frame(self._manus_frames["Right"])
            need_left_manus = self.hand_side == "bimanual" or require_left_control
            left_manus = (
                _copy_frame(self._manus_frames["Left"])
                if need_left_manus
                else None
            )
            ergonomics = {"Right": self._manus_ergonomics["Right"].copy()}
            if need_left_manus:
                ergonomics["Left"] = self._manus_ergonomics["Left"].copy()
            source_times = {
                "manus_right": self._manus_times["Right"],
            }
            if use_vive:
                source_times["vive_right"] = self._vive_right_time
            if need_left_manus:
                source_times["manus_left"] = self._manus_times["Left"]
            if use_vive and self.hand_side == "bimanual":
                source_times.update(
                    {
                        "vive_left": self._vive_left_time,
                    }
                )

        right_frame = (
            reparent_frame(right_manus, vive_right)
            if use_vive
            else right_manus
        )
        if left_manus is None:
            left_frame = None
        elif use_vive and self.hand_side == "bimanual":
            left_frame = reparent_frame(left_manus, vive_left)
        else:
            left_frame = left_manus
        result = {
            "Left": left_frame,
            "Right": right_frame,
            "ergonomics": ergonomics,
            "time": now_wall,
        }

        if self.save_event.is_set():
            with self.lock:
                if self.data is not None:
                    self.data["time"].append(now_wall)
                    self.data["Left"].append(
                        None if left_frame is None else _copy_frame(left_frame)
                    )
                    self.data["Right"].append(_copy_frame(right_frame))
                    if use_vive:
                        self.data["vive_right_time"].append(source_times["vive_right"])
                    self.data["manus_right_time"].append(source_times["manus_right"])
                    if need_left_manus:
                        self.data["manus_left_time"].append(source_times["manus_left"])
                    if use_vive and self.hand_side == "bimanual":
                        self.data["vive_left_time"].append(
                            source_times["vive_left"]
                        )

        return result

    def get_state(self):
        with self.lock:
            return self._pause_state

    def start(self, save_path):
        use_vive = getattr(self, "use_vive", True)
        use_manus = getattr(self, "use_manus", True)
        require_left_control = getattr(self, "require_left_control", True)
        with self.lock:
            self.save_path = save_path
            self.data = {
                "time": [],
                "Left": [],
                "Right": [],
            }
            if use_vive:
                self.data["vive_right_time"] = []
            if use_manus:
                self.data["manus_right_time"] = []
            if use_manus and (self.hand_side == "bimanual" or require_left_control):
                self.data["manus_left_time"] = []
            if use_vive and self.hand_side == "bimanual":
                self.data["vive_left_time"] = []
            self.save_event.set()

    def stop(self):
        use_vive = getattr(self, "use_vive", True)
        use_manus = getattr(self, "use_manus", True)
        require_left_control = getattr(self, "require_left_control", True)
        with self.lock:
            self.save_event.clear()
            save_path = self.save_path
            data = self.data
            metadata = {
                "mode": (
                    ("vive_only_bimanual" if self.hand_side == "bimanual" else "vive_only_unimanual_right")
                    if use_vive and not use_manus
                    else (
                        "vive_manus_bimanual"
                        if self.hand_side == "bimanual"
                        else "vive_manus_unimanual_right"
                    )
                ),
                "use_vive": use_vive,
                "use_manus": use_manus,
                "vive_frame_id": self._vive_frame_id,
                "manus_glove_ids": self._manus_glove_ids.copy(),
                "max_age_s": self.max_age_s,
            }
            if use_manus:
                metadata["manus_topics"] = list(self.manus_topics)
            if use_vive:
                metadata["vive_right_topic"] = self.vive_right_topic
            if use_vive and self.hand_side == "bimanual":
                metadata.update(
                    {
                        "vive_left_topic": self.vive_left_topic,
                        "vive_left_frame_id": self._vive_left_frame_id,
                    }
                )
            self.save_path = None
            self.data = None

        if save_path is None or data is None:
            return
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "time.npy"), np.asarray(data["time"]))
        np.save(
            os.path.join(save_path, "left.npy"),
            np.asarray(data["Left"], dtype=object),
        )
        np.save(
            os.path.join(save_path, "right.npy"),
            np.asarray(data["Right"], dtype=object),
        )
        timestamp_names = []
        if use_vive:
            timestamp_names.append("vive_right_time")
        if use_manus:
            timestamp_names.append("manus_right_time")
        if use_manus and (self.hand_side == "bimanual" or require_left_control):
            timestamp_names.append("manus_left_time")
        if use_vive and self.hand_side == "bimanual":
            timestamp_names.append("vive_left_time")
        for name in timestamp_names:
            np.save(os.path.join(save_path, f"{name}.npy"), np.asarray(data[name]))
        with open(os.path.join(save_path, "metadata.json"), "w") as stream:
            json.dump(metadata, stream, indent=2)

    def end(self):
        if self.save_event.is_set():
            self.stop()
        self.exit_event.set()
        self.executor.shutdown()
        self.spin_thread.join(timeout=2.0)
        try:
            self.executor.remove_node(self.node)
        except Exception:
            pass
        self.node.destroy_node()
        if self._owns_rclpy and self._rclpy.ok():
            self._rclpy.shutdown()

    def is_error(self):
        return self.error_event.is_set()
