#!/usr/bin/env python3
"""
Reset OpenArm to calibration-safe default configuration.

Sequence (all targets are 0.0):
1) left 7, 2) left 6, 3) left 5, 4) left 4, 5) left 3, 6) left 2, 7) left 1,
8) right 4, 9) right 3, 10) right 2, 11) right 1
"""

from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class OpenArmResetNode(Node):
    def __init__(self, time_from_start_sec: int, settle_sec: float) -> None:
        super().__init__("openarm_handeye_reset")
        self.time_from_start_sec = int(time_from_start_sec)
        self.settle_sec = float(settle_sec)
        self.arm_pub = self.create_publisher(
            JointTrajectory, "/openarm/arm_controller/joint_trajectory", 10
        )
        self.sequence = [
            ("openarm_left_joint2", -1.3),
            ("openarm_left_joint3", 0.0),
            ("openarm_left_joint1", 0.0),
            ("openarm_left_joint4", 0.0),
            ("openarm_left_joint7", 0.0),
            ("openarm_left_joint6", 0.0),
            ("openarm_left_joint5", 0.0),
            ("openarm_left_joint2", 0.0),
            ("openarm_right_joint4", 0.0),
            ("openarm_right_joint3", 0.0),
            ("openarm_right_joint1", -0.2),
            ("openarm_right_joint2", 0.15),
            ("openarm_right_joint1", 0.0),
            ("openarm_right_joint2", 0.0),
        ]

    def _publish_single_joint(self, joint_name: str, position: float) -> None:
        msg = JointTrajectory()
        msg.joint_names = [joint_name]
        point = JointTrajectoryPoint()
        point.positions = [float(position)]
        point.time_from_start.sec = self.time_from_start_sec
        msg.points.append(point)
        self.arm_pub.publish(msg)
        self.get_logger().info(
            f"Reset command: {joint_name} -> {position:.3f} (tfs={self.time_from_start_sec}s)"
        )

    def _sleep_with_spin(self, duration_sec: float) -> None:
        end = time.monotonic() + max(0.0, duration_sec)
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def run(self) -> None:
        # Give DDS discovery a brief warm-up.
        self._sleep_with_spin(0.5)
        self.get_logger().info("Starting reset sequence.")
        for joint_name, position in self.sequence:
            self._publish_single_joint(joint_name, position)
            self._sleep_with_spin(self.time_from_start_sec + self.settle_sec)
        self.get_logger().info("Reset sequence completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-from-start-sec", type=int, default=4)
    parser.add_argument("--settle-sec", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.time_from_start_sec <= 0:
        raise ValueError("--time-from-start-sec must be > 0")
    if args.settle_sec < 0:
        raise ValueError("--settle-sec must be >= 0")

    rclpy.init()
    node = OpenArmResetNode(
        time_from_start_sec=args.time_from_start_sec,
        settle_sec=args.settle_sec,
    )
    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
