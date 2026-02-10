#!/usr/bin/env python3
"""
OpenArm handeye capture script.

Flow:
1) Run pre-position sequence (same behavior as standalone replay node).
2) Ask marker confirmation.
3) Replay *_qpos.npy.
4) During replay interval wait (default 2s), capture images per step.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from paradex.calibration.utils import get_handeye_calib_traj, handeye_calib_path_openarm, save_current_camparam
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.utils.file_io import remove_home


def parse_qpos_file_index(path: Path) -> int:
    stem = path.stem  # e.g. "12_qpos"
    if not stem.endswith("_qpos"):
        raise ValueError(f"Unexpected qpos file name: {path.name}")
    return int(stem.split("_")[0])


def collect_qpos_files(qpos_dir: Path, start_idx: int, end_idx: int) -> list[Path]:
    candidates = [p for p in qpos_dir.iterdir() if p.is_file() and p.name.endswith("_qpos.npy")]
    selected = []
    for p in candidates:
        try:
            idx = parse_qpos_file_index(p)
        except ValueError:
            continue
        if start_idx <= idx <= end_idx:
            selected.append((idx, p))
    selected.sort(key=lambda x: x[0])
    return [p for _, p in selected]


class OpenArmQposCaptureNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("openarm_handeye_capture")

        self.qpos_dir = Path(args.qpos_dir).expanduser()
        self.start_idx = int(args.start_idx)
        self.end_idx = int(args.end_idx)
        self.interval_sec = float(args.interval_sec)
        self.root_dir = args.root_dir
        self.rcc = args.rcc

        self.arm_pub = self.create_publisher(
            JointTrajectory, "/openarm/arm_controller/joint_trajectory", 10
        )

        self.arm_joints = [
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

        self._joint_index = {name: i for i, name in enumerate(self.arm_joints)}
        self._right_fixed = {
            "openarm_right_joint1": -0.05,
            "openarm_right_joint2": 1.3,
            "openarm_right_joint3": 1.4,
            "openarm_right_joint4": 1.56,
            "openarm_right_joint5": 0.0,
            "openarm_right_joint6": 0.0,
            "openarm_right_joint7": 0.0,
        }
        # /openarm/joint_states(16) -> arm_joints(14)
        # [head_pitch, head_yaw, r1, r4, l6, l5, l2, l7, l4, r5, l1, r6, r3, l3, r7, r2]
        self._joint_states_to_arm_idx = [10, 6, 13, 8, 5, 4, 7, 2, 15, 12, 3, 9, 11, 14]

        self.file_list = collect_qpos_files(self.qpos_dir, self.start_idx, self.end_idx)
        if not self.file_list:
            raise ValueError(
                f"No qpos files found in {self.qpos_dir} for range [{self.start_idx}, {self.end_idx}]"
            )

    def _confirm_marker_ready(self) -> bool:
        if not sys.stdin.isatty():
            self.get_logger().error("stdin is not a TTY; marker confirmation cannot proceed.")
            return False
        while True:
            answer = input("Marker를 붙였나요? [y/n]: ").strip().lower()
            if answer == "y":
                return True
            if answer == "n":
                return False
            print("y 또는 n으로 입력해주세요.")

    def _publish_single_joint(self, joint_name: str, position: float, tfs_sec: int = 5) -> None:
        msg = JointTrajectory()
        msg.joint_names = [joint_name]
        point = JointTrajectoryPoint()
        point.positions = [float(position)]
        point.time_from_start.sec = int(tfs_sec)
        msg.points.append(point)
        self.arm_pub.publish(msg)
        self.get_logger().info(f"Pre-position published: {joint_name} -> {position:.3f}")

    def _extract_arm_qpos(self, qpos: np.ndarray) -> np.ndarray:
        if qpos.ndim != 1:
            raise ValueError(f"qpos must be 1D, got shape={qpos.shape}")
        if qpos.shape[0] == 14:
            return qpos.astype(np.float64, copy=False)
        if qpos.shape[0] == 16:
            return qpos[self._joint_states_to_arm_idx].astype(np.float64, copy=False)
        raise ValueError(
            f"Unsupported qpos length={qpos.shape[0]}. Expected 14 or 16."
        )

    def _build_replay_positions(self, arm_from_file: np.ndarray) -> np.ndarray:
        q = np.array(arm_from_file, dtype=np.float64, copy=True)
        for joint_name, value in self._right_fixed.items():
            q[self._joint_index[joint_name]] = value
        return q

    def _publish_arm_positions(self, arm_qpos: np.ndarray, file_name: str, tfs_sec: int) -> None:
        msg = JointTrajectory()
        msg.joint_names = self.arm_joints
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in arm_qpos]
        point.time_from_start.sec = int(tfs_sec)
        msg.points.append(point)
        self.arm_pub.publish(msg)
        self.get_logger().info(f"Published {file_name} (tfs={tfs_sec}s)")

    def _capture_step(self, step_idx: int, arm_qpos: np.ndarray) -> None:
        step_dir = os.path.join(self.root_dir, str(step_idx))
        os.makedirs(os.path.join(step_dir, "images"), exist_ok=True)
        np.save(os.path.join(step_dir, "qpos.npy"), arm_qpos)
        self.rcc.start("image", False, remove_home(step_dir))
        self.rcc.stop()
        self.get_logger().info(f"Captured images for step {step_idx}")

    def _sleep_with_spin(self, duration_sec: float) -> None:
        end = time.monotonic() + max(0.0, duration_sec)
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def run(self) -> None:
        # Give DDS discovery a short warm-up period.
        self._sleep_with_spin(0.5)

        # Pre-position sequence: keep behavior before replaying 1_qpos.npy.
        pre_events = [
            ("openarm_left_joint2", -1.3),
            ("openarm_left_joint4", 1.56),
            ("openarm_right_joint1", -0.05),
            ("openarm_right_joint2", 0.2),
            ("openarm_right_joint1", -0.2),
            ("openarm_right_joint2", 1.3),
            ("openarm_right_joint1", -0.05),
            ("openarm_right_joint3", 1.4),
            ("openarm_right_joint4", 1.56),
        ]
        self.get_logger().info(
            "Starting pre-position sequence (5s move per command, next command after 6s)."
        )
        for i, (joint_name, target) in enumerate(pre_events):
            self._publish_single_joint(joint_name, target, tfs_sec=5)
            if i < len(pre_events) - 1:
                self._sleep_with_spin(6.0)
        self._sleep_with_spin(5.0)

        if not self._confirm_marker_ready():
            self.get_logger().info("Marker confirmation failed. Stopping capture.")
            return

        self.get_logger().info(
            f"Replay started. {len(self.file_list)} files, interval capture wait={self.interval_sec:.2f}s."
        )
        for step_idx, file_path in enumerate(self.file_list):
            qpos = np.load(file_path)
            arm_from_file = self._extract_arm_qpos(np.asarray(qpos))
            arm_qpos = self._build_replay_positions(arm_from_file)

            # First replay command uses slower motion, others use 1s.
            tfs_sec = 5 if step_idx == 0 else 1
            self._publish_arm_positions(arm_qpos, file_path.name, tfs_sec=tfs_sec)

            # Wait for motion to finish, then capture during interval wait.
            self._sleep_with_spin(float(tfs_sec + 1))
            interval_start = time.monotonic()
            self._capture_step(step_idx, arm_qpos)
            elapsed = time.monotonic() - interval_start
            self._sleep_with_spin(max(0.0, self.interval_sec - elapsed))

        self.get_logger().info("Replay and capture finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="openarm")
    parser.add_argument("--qpos-dir", type=str, default="/home/temp_id/paradex/system/current/hecalib/openarm/")
    parser.add_argument("--start-idx", type=int, default=1)
    parser.add_argument("--end-idx", type=int, default=30)
    parser.add_argument("--interval-sec", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end_idx < args.start_idx:
        raise ValueError("--end-idx must be >= --start-idx")

    if args.qpos_dir is None:
        args.qpos_dir = get_handeye_calib_traj(args.arm)

    root_dir = os.path.join(handeye_calib_path_openarm, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(root_dir, exist_ok=True)

    rcc = remote_camera_controller("handeye_calibration")
    save_current_camparam(os.path.join(root_dir, "0"))

    args.root_dir = root_dir
    args.rcc = rcc

    rclpy.init()
    node = OpenArmQposCaptureNode(args)
    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        rcc.end()


if __name__ == "__main__":
    main()


    
