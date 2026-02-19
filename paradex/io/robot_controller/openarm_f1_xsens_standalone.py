#!/usr/bin/env python3
"""
Standalone XSens teleop for OpenArm + Inspire RH56F1 hands.

OpenArm (arms/head) is controlled via OpenArmXSensController.
Inspire RH56F1 hands are commanded via ROS2 position controller topics.
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rrc.controllers.openarm_xsens_controller import OpenArmXSensController


def xsens_hand_to_inspire_f1(hand_pose_frame: Dict[str, np.ndarray], is_right: bool = False) -> np.ndarray | None:
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
            # tip_direction[1] *= -1
            # tip_direction[2] *= -1
            # if tip_direction[0] > 0:
            #     # inspire_angles[5] = 1350 + np.arctan(-tip_direction[2] / abs(tip_direction[0])) / np.pi * 300
            #     inspire_angles[5] = np.arcsin(-tip_direction[2]) / np.pi * 2000  * 3.5 - 1000
            #     inspire_angles[4] = -np.arccos(tip_direction[0]) * 800 + 1500
            # else:
            #     # inspire_angles[5] = 1350 + np.arctan(-tip_direction[2] / abs(tip_direction[0])) / np.pi * 300
            #     inspire_angles[5] = np.arccos(-tip_direction[1]) * 2000 - 1000 # no divide by pi for better range
            #     inspire_angles[4] = 300
            if is_right:
                # print(tip_direction, np.arctan(tip_direction[2] / abs(tip_direction[0])))

                inspire_angles[5] = 660 * tip_direction[0] + 700
                if tip_direction[0] < 0:
                    # 0.6, 1.4 -> 1100, 1350
                    # 0.8, 1.4 -> 1100, 1200
                    # inspire_angles[4] = 300 * np.arctan(tip_direction[2] / abs(tip_direction[0])) + 930
                    inspire_angles[4] = 165 * np.arctan(tip_direction[2] / abs(tip_direction[0])) + 968
                else: 
                    # -1.3, -0.1 -> 1100, 1350
                    # inspire_angles[4] = 200 * np.arctan(tip_direction[2] / abs(tip_direction[0])) + 1370          
                    inspire_angles[4] = 1350
            else:
                inspire_angles[5] = -500 * tip_direction[0] + 850
                if tip_direction[0] > 0:
                    # -0.4, -1.4
                    # -0.6, -1.4 -> 1100, 1200
                    # inspire_angles[4] = -250 * np.arctan(tip_direction[2] / abs(tip_direction[0])) + 1000
                    inspire_angles[4] = -125 * np.arctan(tip_direction[2] / abs(tip_direction[0])) + 1025
                else: 
                    # -1.3, -0.1 -> 1100, 1350
                    # inspire_angles[4] = 200 * np.arctan(tip_direction[2] / abs(tip_direction[0])) + 1370          
                    inspire_angles[4] = 1350


            # inspire_angles[4] = 1350

    inspire_angles = np.clip(np.rint(inspire_angles), 0, 1740).astype(np.int32)
    inspire_angles[5] = np.clip(inspire_angles[5], 600, 1800)
    inspire_angles[4] = np.clip(inspire_angles[4], 1100, 1350)

    
    return inspire_angles


class OpenArmF1XSensStandaloneNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("openarm_f1_xsens_standalone")

        self.controller = OpenArmXSensController(
            xsens_port=args.xsens_port,
            scale=args.scale,
            use_keyboard=args.use_keyboard,
            use_recorded=args.use_recorded,
            keyboard_step=args.keyboard_step,
            enable_right_hand=args.enable_right,
        )

        self.head_pub = self.create_publisher(
            JointTrajectory, "/openarm/head_controller/joint_trajectory", 10
        )
        self.arm_pub = self.create_publisher(
            JointTrajectory, "/openarm/arm_controller/joint_trajectory", 10
        )
        self.f1_left_action_pub = self.create_publisher(
            Float64MultiArray, args.left_hand_command_topic, 10
        )
        self.f1_right_action_pub = None
        if args.enable_right:
            self.f1_right_action_pub = self.create_publisher(
                Float64MultiArray, args.right_hand_command_topic, 10
            )

        self.head_joints = ["openarm_head_pitch", "openarm_head_yaw"]
        self.arm_joints = [
            "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
            "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6", "openarm_left_joint7",
            "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
            "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6", "openarm_right_joint7",
        ]

        self._actuator_names = list(self.controller.robot_cfg.actuators.keys())
        self.timer = self.create_timer(1.0 / args.rate_hz, self.loop)
        self.get_logger().info("Started OpenArm XSens + F1 hand standalone teleop.")

    def loop(self) -> None:
        action = self.controller.get_action({})
        joint_map = self._values_to_joint_map(action.values)

        self._pub_head(joint_map)
        self._pub_arms(joint_map)
        self._send_f1_hands_from_xsens()

    def _values_to_joint_map(self, values) -> Dict[str, float]:
        return {name: float(values[i]) for i, name in enumerate(self._actuator_names)}

    def _pub_head(self, joint_map: Dict[str, float]) -> None:
        msg = JointTrajectory()
        msg.joint_names = self.head_joints
        pt = JointTrajectoryPoint()
        pt.positions = [joint_map.get(j, 0.0) for j in self.head_joints]
        pt.time_from_start.sec = 1
        msg.points.append(pt)
        self.head_pub.publish(msg)

    def _pub_arms(self, joint_map: Dict[str, float]) -> None:
        msg = JointTrajectory()
        msg.joint_names = self.arm_joints
        pt = JointTrajectoryPoint()
        pt.positions = [joint_map.get(j, 0.0) for j in self.arm_joints]
        pt.time_from_start.sec = 1
        msg.points.append(pt)
        self.arm_pub.publish(msg)

    def _send_f1_hands_from_xsens(self) -> None:
        cmd_data = self.controller.receiver.get_data()
        if not cmd_data:
            return

        left_hand = cmd_data.get("Left")
        if left_hand is not None:
            left_action = xsens_hand_to_inspire_f1(left_hand, is_right=False)
            if left_action is not None:
                self._publish_hand_command(self.f1_left_action_pub, left_action)

        if self.f1_right_action_pub is not None:
            right_hand = cmd_data.get("Right")
            if right_hand is not None:
                right_action = xsens_hand_to_inspire_f1(right_hand, is_right=True)
                if right_action is not None:
                    self._publish_hand_command(self.f1_right_action_pub, right_action)

    def _publish_hand_command(self, pub, action: np.ndarray) -> None:
        msg = Float64MultiArray()
        # xsens_hand_to_inspire_f1 output:
        # [little, ring, middle, index, thumb_1, thumb_2]
        # Command order:
        # [thumb_1, thumb_2, index, middle, ring, little]
        msg.data = [
            float(action[5]),
            float(action[4]),
            float(action[3]),
            float(action[2]),
            float(action[1]),
            float(action[0]),
        ]
        pub.publish(msg)

    def destroy_node(self) -> bool:
        try:
            self.controller.close()
        finally:
            return super().destroy_node()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--xsens-port", type=int, default=9763)
    p.add_argument("--use-keyboard", action="store_true")
    p.add_argument("--use-recorded", action="store_true")
    p.add_argument("--enable-right", action="store_true")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--keyboard-step", type=float, default=0.02)
    p.add_argument("--rate-hz", type=float, default=50.0)
    p.add_argument("--left-hand-command-topic", type=str, default="/left/position_controller/commands")
    p.add_argument("--right-hand-command-topic", type=str, default="/right/position_controller/commands")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = OpenArmF1XSensStandaloneNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
