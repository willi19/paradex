import json
import os
import time
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
_VIVE_TRACKER_TO_WRIST_ROTATION = np.diag([-1.0, -1.0, -1.0])


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
    """Rotate the tracker frame 180 degrees about its local upward (Y) axis."""
    corrected = transform.copy()
    corrected[:3, :3] = (
        corrected[:3, :3] @ _VIVE_TRACKER_TO_WRIST_ROTATION
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
    """Fuse a right VIVE wrist with MANUS hands using ROS 2 topics."""

    def __init__(
        self,
        vive_right_topic="/vive_trackers/right_hand/pose",
        manus_topics=("/manus_glove_0", "/manus_glove_1"),
        max_age_s=0.25,
    ):
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped
            from manus_ros2_msgs.msg import ManusGlove
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
        except ImportError as exc:
            raise RuntimeError(
                "VIVE teleop requires ROS 2 Humble and the vive-teleop workspace. "
                "Source /home/temp_id/vive-teleop/ros2_ws/install/setup.bash first."
            ) from exc

        self.vive_right_topic = str(vive_right_topic)
        self.manus_topics = tuple(str(topic) for topic in manus_topics)
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
        self._manus_frames = {"Left": None, "Right": None}
        self._manus_times = {"Left": None, "Right": None}
        self._manus_glove_ids = {"Left": None, "Right": None}
        self._pause_state = None
        self.save_path = None
        self.data = None

        node_name = f"vive_manus_teleop_receiver_{os.getpid()}"
        self.node = rclpy.create_node(node_name)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.subscriptions = [
            self.node.create_subscription(
                PoseStamped,
                self.vive_right_topic,
                self._vive_callback,
                qos_profile_sensor_data,
            )
        ]
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
        self.node.get_logger().info(
            "VIVE/MANUS teleop: "
            f"right wrist={self.vive_right_topic}, "
            f"gloves={', '.join(self.manus_topics)}"
        )

    def _spin(self):
        try:
            self.executor.spin()
        except Exception as exc:
            if not self.exit_event.is_set():
                self.node.get_logger().error(f"ROS 2 spin failed: {exc}")
                self.error_event.set()

    def _vive_callback(self, message):
        try:
            transform = apply_vive_tracker_mount_rotation(
                pose_to_matrix(message.pose)
            )
        except (AttributeError, TypeError, ValueError) as exc:
            self.node.get_logger().warning(f"Ignoring invalid VIVE pose: {exc}")
            return

        with self.lock:
            self._vive_right = transform
            self._vive_right_time = time.monotonic()
            self._vive_frame_id = str(message.header.frame_id)

    def _manus_callback(self, message, source_topic):
        side = str(message.side).strip().lower().capitalize()
        if side not in self._manus_frames:
            self.node.get_logger().warning(
                f"Ignoring MANUS message with invalid side '{message.side}' "
                f"from {source_topic}"
            )
            return

        try:
            frame = manus_message_to_frame(message)
        except (AttributeError, TypeError, ValueError) as exc:
            self.node.get_logger().warning(
                f"Ignoring invalid {side} MANUS frame from {source_topic}: {exc}"
            )
            return

        with self.lock:
            self._manus_frames[side] = frame
            self._manus_times[side] = time.monotonic()
            self._manus_glove_ids[side] = int(message.glove_id)
            if side == "Left":
                self._pause_state = _pause_state_from_ergonomics(
                    message.ergonomics
                )

    def get_data(self):
        now_monotonic = time.monotonic()
        now_wall = time.time()
        with self.lock:
            timestamps = (
                self._vive_right_time,
                self._manus_times["Right"],
                self._manus_times["Left"],
            )
            ready = all(timestamp is not None for timestamp in timestamps)
            fresh = ready and all(
                now_monotonic - timestamp <= self.max_age_s
                for timestamp in timestamps
            )
            if not fresh:
                return {"Left": None, "Right": None, "time": now_wall}

            vive_right = self._vive_right.copy()
            right_manus = _copy_frame(self._manus_frames["Right"])
            left_manus = _copy_frame(self._manus_frames["Left"])
            source_times = {
                "vive_right": self._vive_right_time,
                "manus_right": self._manus_times["Right"],
                "manus_left": self._manus_times["Left"],
            }

        right_frame = reparent_frame(right_manus, vive_right)
        result = {
            "Left": left_manus,
            "Right": right_frame,
            "time": now_wall,
        }

        if self.save_event.is_set():
            with self.lock:
                if self.data is not None:
                    self.data["time"].append(now_wall)
                    self.data["Left"].append(_copy_frame(left_manus))
                    self.data["Right"].append(_copy_frame(right_frame))
                    self.data["vive_right_time"].append(source_times["vive_right"])
                    self.data["manus_right_time"].append(source_times["manus_right"])
                    self.data["manus_left_time"].append(source_times["manus_left"])

        return result

    def get_state(self):
        with self.lock:
            return self._pause_state

    def start(self, save_path):
        with self.lock:
            self.save_path = save_path
            self.data = {
                "time": [],
                "Left": [],
                "Right": [],
                "vive_right_time": [],
                "manus_right_time": [],
                "manus_left_time": [],
            }
            self.save_event.set()

    def stop(self):
        with self.lock:
            self.save_event.clear()
            save_path = self.save_path
            data = self.data
            metadata = {
                "mode": "vive_manus_unimanual_right",
                "vive_right_topic": self.vive_right_topic,
                "manus_topics": list(self.manus_topics),
                "vive_frame_id": self._vive_frame_id,
                "manus_glove_ids": self._manus_glove_ids.copy(),
                "max_age_s": self.max_age_s,
            }
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
        for name in ("vive_right_time", "manus_right_time", "manus_left_time"):
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
