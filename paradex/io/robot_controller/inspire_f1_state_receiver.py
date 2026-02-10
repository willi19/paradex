#!/usr/bin/env python3

import os
import time
import argparse
from threading import Event, Lock, Thread
from typing import Dict, List, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from control_msgs.msg import DynamicInterfaceGroupValues


class InspireF1StateReceiver(Node):
    """
    Passive state receiver for Inspire F1 (ROS2).
    - Subscribes to joint_states, position_controller/commands, tactile_sensor_states.
    - Provides InspireF1Controller-like logging interface for CaptureSession.
    """

    def __init__(self, hand_side: str = "both", namespace: str = ""):
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False

        super().__init__("inspire_f1_state_receiver")

        # Avoid clashing with Node.executor internals.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)

        self.hand_sides = self._parse_hand_sides(hand_side)
        self.save_path = None

        self.lock = Lock()
        self.save_event = Event()
        self.error_event = Event()
        self.exit_event = Event()

        self.latest: Dict[str, Dict[str, Optional[object]]] = {}
        for side in self.hand_sides:
            self.latest[side] = {
                "time": None,
                "joint_states": None,
                "commands": None,
                "tactile": None,
            }

        self.data = None

        prefix = f"/{namespace.strip('/')}" if namespace else ""
        self.subs = []
        for side in self.hand_sides:
            side_prefix = f"{prefix}/{side}"
            topic_joint = f"{side_prefix}/joint_states"
            topic_cmd = f"{side_prefix}/position_controller/commands"
            topic_tactile = f"{side_prefix}/tactile_sensor_states"

            self.subs.append(
                self.create_subscription(
                    JointState, topic_joint, lambda msg, s=side: self._joint_cb(s, msg), 50
                )
            )
            self.subs.append(
                self.create_subscription(
                    Float64MultiArray, topic_cmd, lambda msg, s=side: self._cmd_cb(s, msg), 50
                )
            )
            self.subs.append(
                self.create_subscription(
                    DynamicInterfaceGroupValues, topic_tactile, lambda msg, s=side: self._tactile_cb(s, msg), 50
                )
            )

        self.spin_thread = Thread(target=self._spin, daemon=True)
        self.spin_thread.start()

        self.get_logger().info(
            "Subscribed to: " + ", ".join(self._list_topics(prefix))
        )

    def _spin(self):
        try:
            self._executor.spin()
        except Exception as exc:
            self.get_logger().error(f"Spin failed: {exc}")
            self.error_event.set()

    def _parse_hand_sides(self, hand_side: str) -> List[str]:
        if hand_side == "both":
            return ["left", "right"]
        if hand_side in ("left", "right"):
            return [hand_side]
        raise ValueError("hand_side must be 'left', 'right', or 'both'")

    def _list_topics(self, prefix: str) -> List[str]:
        topics = []
        for side in self.hand_sides:
            side_prefix = f"{prefix}/{side}"
            topics.extend(
                [
                    f"{side_prefix}/joint_states",
                    f"{side_prefix}/position_controller/commands",
                    f"{side_prefix}/tactile_sensor_states",
                ]
            )
        return topics

    def _joint_cb(self, side: str, msg: JointState):
        with self.lock:
            self.latest[side]["joint_states"] = np.array(msg.position, dtype=np.float64)
            now = time.time()
            self.latest[side]["time"] = now
            if self.save_event.is_set() and self.data is not None:
                self.data[f"{side}_joint_states_time"].append(now)
                self.data[f"{side}_joint_states"].append(self.latest[side]["joint_states"].copy())

    def _cmd_cb(self, side: str, msg: Float64MultiArray):
        with self.lock:
            self.latest[side]["commands"] = np.array(msg.data, dtype=np.float64)
            now = time.time()
            self.latest[side]["time"] = now
            if self.save_event.is_set() and self.data is not None:
                self.data[f"{side}_commands_time"].append(now)
                self.data[f"{side}_commands"].append(self.latest[side]["commands"].copy())

    def _tactile_cb(self, side: str, msg: DynamicInterfaceGroupValues):
        group_name = f"{side}_hand_tactile_sensor"
        tactile_map = {}
        for i, g in enumerate(msg.interface_groups):
            if g == group_name and i < len(msg.interface_values):
                iv = msg.interface_values[i]
                for name, val in zip(iv.interface_names, iv.values):
                    tactile_map[name] = val
                break
        if tactile_map:
            with self.lock:
                self.latest[side]["tactile"] = tactile_map
                now = time.time()
                self.latest[side]["time"] = now
                if self.save_event.is_set() and self.data is not None:
                    self.data[f"{side}_tactile_time"].append(now)
                    self.data[f"{side}_tactile"].append(dict(tactile_map))

    def start(self, save_path: str):
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        with self.lock:
            self.data = {}
            for side in self.hand_sides:
                self.data[f"{side}_joint_states_time"] = []
                self.data[f"{side}_joint_states"] = []
                self.data[f"{side}_commands_time"] = []
                self.data[f"{side}_commands"] = []
                self.data[f"{side}_tactile_time"] = []
                self.data[f"{side}_tactile"] = []
            self.save_event.set()

        self.get_logger().info(f"Started logging → {self.save_path}")

    def stop(self):
        with self.lock:
            data = self.data
            save_path = self.save_path
            self.data = None
            self.save_path = None
            self.save_event.clear()

        if data is None or save_path is None:
            return

        os.makedirs(save_path, exist_ok=True)
        for name, values in data.items():
            np.save(os.path.join(save_path, f"{name}.npy"), np.array(values, dtype=object))

        first_side = self.hand_sides[0]
        saved = len(data.get(f"{first_side}_joint_states_time", []))
        self.get_logger().info(f"Saved {saved} joint-state frames to {save_path}")

    def end(self):
        self.exit_event.set()

        try:
            self._executor.remove_node(self)
        except Exception:
            pass
        self._executor.shutdown()
        self.destroy_node()
        if self._owns_rclpy:
            rclpy.shutdown()
        self.spin_thread.join(timeout=2.0)

    def move(self, *_, **__):
        self.get_logger().warning("InspireF1StateReceiver.move() called, but no control is implemented.")

    def get_data(self, side: Optional[str] = None):
        side = self.hand_sides[0] if side is None else side
        if side not in self.latest:
            raise ValueError(f"unknown side: {side}")
        with self.lock:
            qpos = (
                None
                if self.latest[side]["joint_states"] is None
                else self.latest[side]["joint_states"].copy()
            )
            tactile = (
                None
                if self.latest[side]["tactile"] is None
                else dict(self.latest[side]["tactile"])
            )
            last_time = self.latest[side]["time"]
        return {
            "qpos": qpos,
            "position": None,
            "tactile": tactile,
            "time": last_time if last_time is not None else time.time(),
        }

    def is_error(self):
        return self.error_event.is_set()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save joint_states/commands/tactile npy files",
    )
    p.add_argument(
        "--hand-side",
        type=str,
        default="both",
        choices=["left", "right", "both"],
        help="Hand side to subscribe (left/right/both)",
    )
    p.add_argument(
        "--namespace",
        type=str,
        default="",
        help="ROS2 namespace prefix (optional)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    node = InspireF1StateReceiver(hand_side=args.hand_side, namespace=args.namespace)
    node.start(args.save_dir)
    try:
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.end()


if __name__ == "__main__":
    main()
