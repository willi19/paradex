import json
import os
import socket
import time
from threading import Event, Lock, Thread
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


QUEST_JOINT_ORDER = [
    "Wrist",
    "Palm",
    "ThumbMetacarpal",
    "ThumbProximal",
    "ThumbDistal",
    "IndexMetacarpal",
    "IndexProximal",
    "IndexIntermediate",
    "IndexDistal",
    "MiddleMetacarpal",
    "MiddleProximal",
    "MiddleIntermediate",
    "MiddleDistal",
    "RingMetacarpal",
    "RingProximal",
    "RingIntermediate",
    "RingDistal",
    "LittleMetacarpal",
    "LittleProximal",
    "LittleIntermediate",
    "LittleDistal",
    "ThumbTip",
    "IndexTip",
    "MiddleTip",
    "RingTip",
    "LittleTip",
]

_QUEST_TO_PARADEX_NAME = {
    "Wrist": "wrist",
    "Palm": "palm",
    "ThumbMetacarpal": "thumb_metacarpal",
    "ThumbProximal": "thumb_proximal",
    "ThumbDistal": "thumb_distal",
    "ThumbTip": "thumb_tip",
    "IndexMetacarpal": "index_metacarpal",
    "IndexProximal": "index_proximal",
    "IndexIntermediate": "index_intermediate",
    "IndexDistal": "index_distal",
    "IndexTip": "index_tip",
    "MiddleMetacarpal": "middle_metacarpal",
    "MiddleProximal": "middle_proximal",
    "MiddleIntermediate": "middle_intermediate",
    "MiddleDistal": "middle_distal",
    "MiddleTip": "middle_tip",
    "RingMetacarpal": "ring_metacarpal",
    "RingProximal": "ring_proximal",
    "RingIntermediate": "ring_intermediate",
    "RingDistal": "ring_distal",
    "RingTip": "ring_tip",
    "LittleMetacarpal": "pinky_metacarpal",
    "LittleProximal": "pinky_proximal",
    "LittleIntermediate": "pinky_intermediate",
    "LittleDistal": "pinky_distal",
    "LittleTip": "pinky_tip",
}

_QUEST_PARENT = {
    "Wrist": None,
    "Palm": "Wrist",
    "ThumbMetacarpal": "Palm",
    "ThumbProximal": "ThumbMetacarpal",
    "ThumbDistal": "ThumbProximal",
    "ThumbTip": "ThumbDistal",
    "IndexMetacarpal": "Palm",
    "IndexProximal": "IndexMetacarpal",
    "IndexIntermediate": "IndexProximal",
    "IndexDistal": "IndexIntermediate",
    "IndexTip": "IndexDistal",
    "MiddleMetacarpal": "Palm",
    "MiddleProximal": "MiddleMetacarpal",
    "MiddleIntermediate": "MiddleProximal",
    "MiddleDistal": "MiddleIntermediate",
    "MiddleTip": "MiddleDistal",
    "RingMetacarpal": "Palm",
    "RingProximal": "RingMetacarpal",
    "RingIntermediate": "RingProximal",
    "RingDistal": "RingIntermediate",
    "RingTip": "RingDistal",
    "LittleMetacarpal": "Palm",
    "LittleProximal": "LittleMetacarpal",
    "LittleIntermediate": "LittleProximal",
    "LittleDistal": "LittleIntermediate",
    "LittleTip": "LittleDistal",
}

# OpenXR/Unity: +X right, +Y up, +Z forward. ParaDex/xArm: +X forward,
# +Y left, +Z up. This is a basis change, so rotations use C R C^T.
QUEST_TO_GLOBAL = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)

_QUEST_WRIST_BASIS = {
    "Left": np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    ),
    "Right": np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    ),
}

# Each legacy hand basis is a reflection because it was defined in Unity's
# left-handed coordinates. Combining it with the global basis reflection gives
# a proper rotation that can safely be used in homogeneous transforms.
_QUEST_TO_WRIST = {
    side: hand_basis @ QUEST_TO_GLOBAL.T
    for side, hand_basis in _QUEST_WRIST_BASIS.items()
}

occulus_hand_joint_name = [
    "palm",
    "wrist",
    "thumb_metacarpal",
    "thumb_proximal",
    "thumb_distal",
    "thumb_tip",
    "index_metacarpal",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "index_tip",
    "middle_metacarpal",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "middle_tip",
    "ring_metacarpal",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "ring_tip",
    "pinky_metacarpal",
    "pinky_proximal",
    "pinky_intermediate",
    "pinky_distal",
    "pinky_tip",
]

occulus_hand_joint_parent_name = {
    _QUEST_TO_PARADEX_NAME[name]: (
        _QUEST_TO_PARADEX_NAME[_QUEST_PARENT[name]]
        if _QUEST_PARENT[name] is not None
        else None
    )
    for name in QUEST_JOINT_ORDER
}

_REQUIRED_TELEOP_JOINTS = set(occulus_hand_joint_name) - {
    "palm",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
}


class Quest3Receiver:
    """Receive Quest hand packets and expose the existing ParaDex pose contract."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9000,
        max_age_s: float = 0.75,
        extrapolation_delay_s: float = 0.03,
        extrapolation_horizon_s: float = 0.10,
        max_linear_speed_m_s: float = 0.5,
        max_angular_speed_rad_s: float = 3.0,
    ) -> None:
        self.host = host
        self.max_age_s = float(max_age_s)
        self.extrapolation_delay_s = max(0.0, float(extrapolation_delay_s))
        self.extrapolation_horizon_s = max(0.0, float(extrapolation_horizon_s))
        self.max_linear_speed_m_s = max(0.0, float(max_linear_speed_m_s))
        self.max_angular_speed_rad_s = max(0.0, float(max_angular_speed_rad_s))
        self.exit_event = Event()
        self.error_event = Event()
        self.lock = Lock()
        self.hand_pose: Dict[str, Dict[str, np.ndarray]] = {}
        self.hand_time: Dict[str, float] = {}
        self.hand_velocity: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.control_state = 0
        self.control_tracked = False
        self._recording = False
        self._recorded_packets = []
        self.save_path: Optional[str] = None

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, int(port)))
        self.port = self.socket.getsockname()[1]
        self.socket.settimeout(0.1)

        self.recv_thread = Thread(target=self._run, daemon=True)
        self.recv_thread.start()
        print("Quest 3 receiver listening on UDP {}:{}".format(host, self.port))

    @staticmethod
    def _safe_rotation(values) -> Optional[np.ndarray]:
        try:
            quaternion = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            return None
        norm = np.linalg.norm(quaternion)
        if norm < 1e-8:
            return None
        return R.from_quat(quaternion / norm).as_matrix()

    def _parse_hand_packet(
        self, packet: Dict
    ) -> Optional[Tuple[str, Dict[str, np.ndarray]]]:
        label = str(packet.get("hand", "")).lower()
        if label not in ("left", "right"):
            return None
        side = label.capitalize()

        quaternions = packet.get("q")
        positions = packet.get("jp")
        mask = packet.get("m")
        if not isinstance(quaternions, list) or not isinstance(positions, list):
            return None

        joint_count = len(quaternions) // 4
        if joint_count not in (21, 26) or len(positions) < joint_count * 3:
            return None

        local_rotations = {}
        joint_positions = {}
        for index, name in enumerate(QUEST_JOINT_ORDER[:joint_count]):
            if isinstance(mask, list) and index < len(mask) and not mask[index]:
                continue
            rotation = self._safe_rotation(quaternions[index * 4 : index * 4 + 4])
            try:
                position = np.asarray(
                    positions[index * 3 : index * 3 + 3], dtype=np.float64
                )
            except (TypeError, ValueError):
                continue
            if rotation is None or position.shape != (3,) or not np.all(np.isfinite(position)):
                continue
            local_rotations[name] = rotation
            joint_positions[name] = position

        global_rotations = {}
        for name in QUEST_JOINT_ORDER[:joint_count]:
            if name not in local_rotations:
                continue
            parent = _QUEST_PARENT[name]
            if parent is None:
                global_rotations[name] = local_rotations[name]
            elif parent in global_rotations:
                global_rotations[name] = global_rotations[parent] @ local_rotations[name]

        wrist_correction_inv = _QUEST_TO_WRIST[side].T
        hand_pose = {}
        for name, global_rotation in global_rotations.items():
            if name not in joint_positions:
                continue
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = (
                QUEST_TO_GLOBAL
                @ global_rotation
                @ QUEST_TO_GLOBAL.T
                @ wrist_correction_inv
            )
            pose[:3, 3] = QUEST_TO_GLOBAL @ joint_positions[name]
            hand_pose[_QUEST_TO_PARADEX_NAME[name]] = pose

        if "wrist" not in hand_pose:
            return None
        return side, hand_pose

    @staticmethod
    def _limit_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if max_norm <= 0.0:
            return np.zeros_like(vector)
        if norm <= max_norm or norm < 1e-12:
            return vector
        return vector * (max_norm / norm)

    @staticmethod
    def _transform_hand_pose(
        hand_pose: Dict[str, np.ndarray], transform: np.ndarray
    ) -> Dict[str, np.ndarray]:
        return {name: transform @ pose for name, pose in hand_pose.items()}

    def _update_hand_pose(
        self, side: str, partial_pose: Dict[str, np.ndarray], sample_time: float
    ) -> bool:
        previous = self.hand_pose.get(side)
        if previous is None:
            if not _REQUIRED_TELEOP_JOINTS.issubset(partial_pose):
                return False
            self.hand_pose[side] = {
                name: pose.copy() for name, pose in partial_pose.items()
            }
            self.hand_time[side] = sample_time
            self.hand_velocity[side] = (np.zeros(3), np.zeros(3))
            return True

        previous_wrist = previous["wrist"]
        current_wrist = partial_pose["wrist"]
        wrist_delta = current_wrist @ np.linalg.inv(previous_wrist)
        merged = self._transform_hand_pose(previous, wrist_delta)
        merged.update({name: pose.copy() for name, pose in partial_pose.items()})

        previous_time = self.hand_time[side]
        dt = sample_time - previous_time
        if 1e-4 < dt <= self.max_age_s:
            linear_velocity = (
                current_wrist[:3, 3] - previous_wrist[:3, 3]
            ) / dt
            angular_velocity = R.from_matrix(
                current_wrist[:3, :3] @ previous_wrist[:3, :3].T
            ).as_rotvec() / dt
            linear_velocity = self._limit_norm(
                linear_velocity, self.max_linear_speed_m_s
            )
            angular_velocity = self._limit_norm(
                angular_velocity, self.max_angular_speed_rad_s
            )
        else:
            linear_velocity = np.zeros(3)
            angular_velocity = np.zeros(3)

        self.hand_pose[side] = merged
        self.hand_time[side] = sample_time
        self.hand_velocity[side] = (linear_velocity, angular_velocity)
        return True

    def _extrapolate_hand_pose(
        self,
        hand_pose: Dict[str, np.ndarray],
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
        age_s: float,
    ) -> Dict[str, np.ndarray]:
        horizon = min(
            max(0.0, age_s - self.extrapolation_delay_s),
            self.extrapolation_horizon_s,
        )
        if horizon <= 0.0:
            return {name: pose.copy() for name, pose in hand_pose.items()}

        wrist = hand_pose["wrist"]
        predicted_wrist = wrist.copy()
        predicted_wrist[:3, 3] += linear_velocity * horizon
        predicted_wrist[:3, :3] = (
            R.from_rotvec(angular_velocity * horizon).as_matrix()
            @ wrist[:3, :3]
        )
        wrist_delta = predicted_wrist @ np.linalg.inv(wrist)
        return self._transform_hand_pose(hand_pose, wrist_delta)

    def _update_control(self, packet: Dict) -> None:
        if "control_tracked" not in packet:
            return
        if packet.get("control_tracked") is not True:
            self.control_tracked = False
            return
        try:
            state = int(packet.get("control_state", 0))
        except (TypeError, ValueError):
            return
        if state in (0, 1, 2, 3):
            self.control_state = state
            self.control_tracked = True

    def _run(self) -> None:
        while not self.exit_event.is_set():
            try:
                payload, _ = self.socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                if not self.exit_event.is_set():
                    self.error_event.set()
                break

            try:
                packet = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(packet, dict):
                continue

            received_at = time.time()
            sample_time = time.monotonic()
            parsed = self._parse_hand_packet(packet)
            with self.lock:
                self._update_control(packet)
                if parsed is not None:
                    side, hand_pose = parsed
                    self._update_hand_pose(side, hand_pose, sample_time)
                if self._recording:
                    record = dict(packet)
                    record["received_at"] = received_at
                    self._recorded_packets.append(record)

    def get_data(self) -> Dict:
        now = time.monotonic()
        with self.lock:
            result = {}
            for side in ("Left", "Right"):
                pose = self.hand_pose.get(side)
                updated_at = self.hand_time.get(side)
                if (
                    pose is None
                    or updated_at is None
                    or now - updated_at > self.max_age_s
                ):
                    result[side] = None
                else:
                    linear_velocity, angular_velocity = self.hand_velocity.get(
                        side, (np.zeros(3), np.zeros(3))
                    )
                    result[side] = self._extrapolate_hand_pose(
                        pose,
                        linear_velocity,
                        angular_velocity,
                        now - updated_at,
                    )
            result["control_state"] = self.control_state
            result["control_tracked"] = self.control_tracked
        result["time"] = time.time()
        return result

    def get_control_state(self) -> int:
        with self.lock:
            return self.control_state

    def start(self, save_path: str) -> None:
        with self.lock:
            self.save_path = save_path
            self._recorded_packets = []
            self._recording = True

    def stop(self) -> None:
        with self.lock:
            if not self._recording or self.save_path is None:
                return
            self._recording = False
            save_path = self.save_path
            packets = list(self._recorded_packets)
            self.save_path = None
            self._recorded_packets = []

        os.makedirs(save_path, exist_ok=True)
        output_path = os.path.join(save_path, "quest_packets.jsonl")
        with open(output_path, "w") as output_file:
            for packet in packets:
                output_file.write(json.dumps(packet, separators=(",", ":")) + "\n")

    def end(self) -> None:
        self.stop()
        self.exit_event.set()
        self.socket.close()
        self.recv_thread.join(timeout=1.0)
        print("Quest 3 receiver terminated")

    def quit(self) -> None:
        self.end()

    def is_error(self) -> bool:
        return self.error_event.is_set()


# Preserve the existing public import and the historical CLI spelling.
OculusReceiver = Quest3Receiver
