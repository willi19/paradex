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


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "t", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "f", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


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

        self.left_qpos_dir = (
            None if args.left_qpos_dir is None else Path(args.left_qpos_dir).expanduser()
        )
        self.right_qpos_dir = (
            None if args.right_qpos_dir is None else Path(args.right_qpos_dir).expanduser()
        )
        self.start_idx = int(args.start_idx)
        self.end_idx = int(args.end_idx)
        self.interval_sec = float(args.interval_sec)
        self.augment_count = int(args.augment_count)
        self.perturb_std = float(args.perturb_std)
        self.perturb_clip = float(args.perturb_clip)
        self.augment_seed = None if args.augment_seed < 0 else int(args.augment_seed)
        self.include_left_hand = bool(args.include_left_hand)
        self.include_right_hand = bool(args.include_right_hand)
        self.root_dir = args.root_dir
        self.left_root_dir = os.path.join(self.root_dir, "left")
        self.right_root_dir = os.path.join(self.root_dir, "right")
        self.rcc = args.rcc
        self._rng = np.random.default_rng(self.augment_seed)

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
        # Symmetric fixed posture for the opposite arm during right-hand replay.
        self._left_fixed = {
            "openarm_left_joint1": 0.05,
            "openarm_left_joint2": -1.3,
            "openarm_left_joint3": -1.4,
            "openarm_left_joint4": 1.56,
            "openarm_left_joint5": 0.0,
            "openarm_left_joint6": 0.0,
            "openarm_left_joint7": 0.0,
        }
        # /openarm/joint_states(16) -> arm_joints(14)
        # [head_pitch, head_yaw, r1, r4, l6, l5, l2, l7, l4, r5, l1, r6, r3, l3, r7, r2]
        self._joint_states_to_arm_idx = [10, 6, 13, 8, 5, 4, 7, 2, 15, 12, 3, 9, 11, 14]

        self.left_file_list: list[Path] = []
        if self.include_left_hand:
            if self.left_qpos_dir is None:
                raise ValueError("--left-qpos-dir must be set when --include-left-hand is enabled.")
            self.left_file_list = collect_qpos_files(
                self.left_qpos_dir, self.start_idx, self.end_idx
            )
            if not self.left_file_list:
                raise ValueError(
                    f"No qpos files found in {self.left_qpos_dir} for range [{self.start_idx}, {self.end_idx}]"
                )
        self.right_file_list: list[Path] = []
        if self.include_right_hand:
            if self.right_qpos_dir is None:
                raise ValueError("--right-qpos-dir must be set when --include-right-hand is enabled.")
            self.right_file_list = collect_qpos_files(
                self.right_qpos_dir, self.start_idx, self.end_idx
            )
            if not self.right_file_list:
                raise ValueError(
                    f"No qpos files found in {self.right_qpos_dir} for range [{self.start_idx}, {self.end_idx}]"
                )

    def _build_replay_plan(
        self,
        file_list: list[Path],
        fixed_map: dict[str, float] | None,
    ) -> list[tuple[str, np.ndarray]]:
        plan: list[tuple[str, np.ndarray]] = []
        arm_from_file_list: list[np.ndarray] = []

        for file_path in file_list:
            qpos = np.load(file_path)
            arm_from_file = self._extract_arm_qpos(np.asarray(qpos))
            arm_from_file_list.append(arm_from_file)
            arm_qpos = self._build_replay_positions(arm_from_file, fixed_map=fixed_map)
            plan.append((file_path.name, arm_qpos))

        if self.augment_count <= 0:
            return plan
        if not arm_from_file_list:
            return plan

        for aug_idx in range(self.augment_count):
            src_idx = int(self._rng.integers(0, len(arm_from_file_list)))
            noise = self._rng.normal(loc=0.0, scale=self.perturb_std, size=arm_from_file_list[src_idx].shape)
            if self.perturb_clip > 0:
                noise = np.clip(noise, -self.perturb_clip, self.perturb_clip)
            augmented = arm_from_file_list[src_idx] + noise
            arm_qpos = self._build_replay_positions(augmented, fixed_map=fixed_map)
            plan.append((f"aug_{aug_idx + 1:03d}_from_{file_list[src_idx].name}", arm_qpos))

        return plan

    def _confirm_marker_ready(self) -> bool:
        if not sys.stdin.isatty():
            self.get_logger().error("stdin is not a TTY; marker confirmation cannot proceed.")
            return False
        while True:
            answer = input("Is the marker attached? [y/n]: ").strip().lower()
            if answer == "y":
                return True
            if answer == "n":
                return False
            print("Please enter 'y' or 'n'.")

    def _confirm_marker_moved_to_right(self) -> bool:
        if not sys.stdin.isatty():
            self.get_logger().error("stdin is not a TTY; marker hand-switch confirmation cannot proceed.")
            return False
        while True:
            answer = input("Move marker to RIGHT hand done? Enter 'y' to continue [y/n]: ").strip().lower()
            if answer == "y":
                return True
            if answer == "n":
                return False
            print("Please enter 'y' or 'n'.")

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

    def _build_replay_positions(
        self, arm_from_file: np.ndarray, fixed_map: dict[str, float] | None = None
    ) -> np.ndarray:
        q = np.array(arm_from_file, dtype=np.float64, copy=True)
        if fixed_map is not None:
            for joint_name, value in fixed_map.items():
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

    def _capture_step(self, side: str, step_idx: int, arm_qpos: np.ndarray) -> None:
        root = self.left_root_dir if side == "left" else self.right_root_dir
        step_dir = os.path.join(root, str(step_idx))
        os.makedirs(os.path.join(step_dir, "images"), exist_ok=True)
        np.save(os.path.join(step_dir, "qpos.npy"), arm_qpos)
        self.rcc.start("image", False, remove_home(step_dir))
        self.rcc.stop()
        self.get_logger().info(f"[{side}] Captured images for step {step_idx}")

    def _sleep_with_spin(self, duration_sec: float) -> None:
        end = time.monotonic() + max(0.0, duration_sec)
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def run_left(self, go_to_starting_position: bool = False) -> None:
        # Give DDS discovery a short warm-up period.
        self._sleep_with_spin(0.5)

        # Pre-position sequence: keep behavior before replaying 1_qpos.npy.
        if go_to_starting_position:
            pre_events = [
                ("openarm_left_joint2", -1.3),
                ("openarm_left_joint4", 1.56),
                ("openarm_right_joint1", -0.05),
                ("openarm_right_joint2", 0.2),
                ("openarm_right_joint1", -0.2),
                ("openarm_right_joint2", 1.3),
                # ("openarm_right_joint1", -0.05),
                ("openarm_right_joint3", 1.4),
                ("openarm_right_joint4", 1.56),
            ]
            self.get_logger().info(
                "Starting pre-position sequence (5s move per command, next command after 6s)."
            )
            for i, (joint_name, target) in enumerate(pre_events):
                if target < 0.5:
                    self._publish_single_joint(joint_name, target, tfs_sec=2)
                    if i < len(pre_events) - 1:
                        self._sleep_with_spin(3.0)
                else:
                    self._publish_single_joint(joint_name, target, tfs_sec=5)
                    if i < len(pre_events) - 1:
                        self._sleep_with_spin(6.0)
            self._sleep_with_spin(5.0)

        if not self._confirm_marker_ready():
            self.get_logger().info("Marker confirmation failed. Stopping capture.")
            return

        left_plan = self._build_replay_plan(self.left_file_list, fixed_map=self._right_fixed)
        self.get_logger().info(
            f"[left] Replay started. base={len(self.left_file_list)}, augmented={max(0, len(left_plan) - len(self.left_file_list))}, total={len(left_plan)}, interval capture wait={self.interval_sec:.2f}s."
        )
        for step_idx, (entry_name, arm_qpos) in enumerate(left_plan):

            # First replay command uses slower motion, others use 1s.
            tfs_sec = 5 if step_idx == 0 else 3
            self._publish_arm_positions(arm_qpos, entry_name, tfs_sec=tfs_sec)

            # Wait for motion to finish, then capture during interval wait.
            self._sleep_with_spin(float(tfs_sec + 2))
            interval_start = time.monotonic()
            self._capture_step("left", step_idx, arm_qpos)
            elapsed = time.monotonic() - interval_start
            self._sleep_with_spin(max(0.0, self.interval_sec - elapsed))

        self.get_logger().info("[left] Replay and capture finished.")

    def run_right(self) -> None:
        if not self.right_file_list:
            self.get_logger().warning("Right qpos list is empty. Skipping right replay.")
            return

        
        pre_events = [
            ("openarm_left_joint2", -1.3),
            # ("openarm_right_joint1", -0.05),
            # ("openarm_right_joint2", 0.2),
            # ("openarm_right_joint1", -0.2),
            # ("openarm_right_joint2", 1.3),
            # ("openarm_right_joint3", 1.4),
            # ("openarm_right_joint4", 1.56),
        ]
        self.get_logger().info(
            "Starting pre-position sequence (5s move per command, next command after 6s)."
        )
        for i, (joint_name, target) in enumerate(pre_events):
            if target < 0.5:
                self._publish_single_joint(joint_name, target, tfs_sec=2)
                if i < len(pre_events) - 1:
                    self._sleep_with_spin(3.0)
            else:
                self._publish_single_joint(joint_name, target, tfs_sec=5)
                if i < len(pre_events) - 1:
                    self._sleep_with_spin(6.0)
        self._sleep_with_spin(5.0)


        right_plan = self._build_replay_plan(self.right_file_list, fixed_map=self._left_fixed)

        self.get_logger().info(
            f"[right] Replay started. base={len(self.right_file_list)}, augmented={max(0, len(right_plan) - len(self.right_file_list))}, total={len(right_plan)}, interval capture wait={self.interval_sec:.2f}s."
        )
        for step_idx, (entry_name, arm_qpos) in enumerate(right_plan):

            tfs_sec = 5 if step_idx == 0 else 1
            self._publish_arm_positions(arm_qpos, entry_name, tfs_sec=tfs_sec)

            self._sleep_with_spin(float(tfs_sec + 1))
            interval_start = time.monotonic()
            self._capture_step("right", step_idx, arm_qpos)
            elapsed = time.monotonic() - interval_start
            self._sleep_with_spin(max(0.0, self.interval_sec - elapsed))

        self.get_logger().info("[right] Replay and capture finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="openarm")
    parser.add_argument("--left-qpos-dir", type=str, default=None)
    parser.add_argument("--right-qpos-dir", type=str, default=None)
    parser.add_argument("--start-idx", type=int, default=1)
    parser.add_argument("--end-idx", type=int, default=100)
    parser.add_argument(
        "--augment-count",
        type=int,
        default=100,
        help="Number of additional replay poses generated by small random joint perturbation.",
    )
    parser.add_argument(
        "--perturb-std",
        type=float,
        default=0.015,
        help="Standard deviation (rad) for Gaussian joint perturbation.",
    )
    parser.add_argument(
        "--perturb-clip",
        type=float,
        default=0.04,
        help="Per-joint perturbation clamp magnitude (rad). Set <=0 to disable clipping.",
    )
    parser.add_argument(
        "--augment-seed",
        type=int,
        default=20260212,
        help="Random seed for perturbation. Use negative value for non-deterministic seed.",
    )
    parser.add_argument("--interval-sec", type=float, default=2.0)
    parser.add_argument("--go-to-starting-position", action="store_true")
    parser.add_argument(
        "--include-left-hand",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--include-right-hand", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end_idx < args.start_idx:
        raise ValueError("--end-idx must be >= --start-idx")

    if args.include_left_hand and args.left_qpos_dir is None:
        args.left_qpos_dir = get_handeye_calib_traj(args.arm, "left")
    if args.include_right_hand and args.right_qpos_dir is None:
        args.right_qpos_dir = get_handeye_calib_traj(args.arm, "right")

    root_dir = os.path.join(handeye_calib_path_openarm, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(root_dir, exist_ok=True)

    rcc = remote_camera_controller("handeye_calibration")
    save_current_camparam(os.path.join(root_dir, "0"))

    args.root_dir = root_dir
    args.rcc = rcc

    rclpy.init()
    node = OpenArmQposCaptureNode(args)
    try:
        if args.include_left_hand:
            node.run_left(args.go_to_starting_position)
        if args.include_right_hand:
            if args.include_left_hand:
                if node._confirm_marker_moved_to_right():
                    node.run_right()
                else:
                    node.get_logger().info("Right-hand marker confirmation failed. Skipping right replay.")
            else:
                node.run_right()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        rcc.end()


if __name__ == "__main__":
    main()


    
