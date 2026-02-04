#!/usr/bin/env python3

import os
import time
import argparse
import numpy as np
from threading import Event, Lock, Thread

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class OpenArmStateReceiver(Node):
    """
    Passive state receiver for OpenArm.
    - Subscribes to joint states.
    - Provides XArmController-like interface for CaptureSession.
    """

    def __init__(self, topic: str = "/openarm/joint_states"):
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False

        super().__init__("openarm_state_receiver")

        self.topic = topic
        self.save_path = None

        self.lock = Lock()
        self.save_event = Event()
        self.error_event = Event()

        self.data = None
        self.last_state = {
            "time": None,
            "position": None,
            # "velocity": None,
            # "torque": None,
        }

        self.sub = self.create_subscription(
            JointState,
            self.topic,
            self._callback,
            100,
        )

        self.spin_thread = Thread(target=self._spin, daemon=True)
        self.spin_thread.start()

        self.get_logger().info(f"Subscribed to {self.topic}")

    def _spin(self):
        try:
            rclpy.spin(self)
        except Exception:
            self.error_event.set()

    def _callback(self, msg: JointState):
        now = time.time()
        position = np.array(msg.position, dtype=np.float64)
        
        ################ Joint Mapping for OpenArm ################
        # OpenArm joint order:
        #['openarm_head_pitch', 'openarm_head_yaw', 'openarm_right_joint1', 
        # 'openarm_right_joint4', 'openarm_left_joint6', 'openarm_left_joint5', 
        # 'openarm_left_joint2', 'openarm_left_joint7', 'openarm_left_joint4', 
        # 'openarm_right_joint5', 'openarm_left_joint1', 'openarm_right_joint6', 
        # 'openarm_right_joint3', 'openarm_left_joint3', 'openarm_right_joint7', 
        # 'openarm_right_joint2']
        
        # print(position)
        # velocity = (
        #     np.array(msg.velocity, dtype=np.float64)
        #     if msg.velocity
        #     else np.zeros_like(position)
        # )
        # torque = (
        #     np.array(msg.effort, dtype=np.float64)
        #     if msg.effort
        #     else np.zeros_like(position)
        # )

        with self.lock:
            self.last_state["time"] = now
            self.last_state["position"] = position
            # self.last_state["velocity"] = velocity
            # self.last_state["torque"] = torque

            if self.save_event.is_set() and self.data is not None:
                self.data["time"].append(now)
                self.data["position"].append(position)
                # self.data["velocity"].append(velocity)
                # self.data["torque"].append(torque)

    def start(self, save_path: str):
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        with self.lock:
            self.data = {
                "time": [],
                "position": [],
                # "velocity": [],
                # "torque": [],
            }
            self.save_event.set()

        self.get_logger().info(f"Started logging → {self.save_path}")

    def stop(self):
        with self.lock:
            self.save_event.clear()
            data = self.data
            save_path = self.save_path
            self.data = None
            self.save_path = None

        if data is None or save_path is None:
            return

        os.makedirs(save_path, exist_ok=True)
        for name, values in data.items():
            if len(values) == 0:
                arr = np.zeros((0,), dtype=np.float64)
            else:
                arr = np.stack(values, axis=0)
            np.save(os.path.join(save_path, f"{name}.npy"), arr)

        self.get_logger().info(
            f"Saved {len(data['time'])} frames to {save_path}"
        )

    def end(self):
        if self.save_event.is_set():
            self.stop()

        self.destroy_node()
        if self._owns_rclpy:
            rclpy.shutdown()
        self.spin_thread.join(timeout=2.0)

    def move(self, *_, **__):
        # Passive receiver: no control.
        self.get_logger().warning("OpenArmStateReceiver.move() called, but no control is implemented.")

    def get_data(self):
        with self.lock:
            qpos = None if self.last_state["position"] is None else self.last_state["position"].copy()
            last_time = self.last_state["time"]

        # No FK available here; return identity for compatibility.
        return {
            "qpos": qpos,
            "position": np.eye(4),
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
        help="Directory to save time/position/velocity/torque npy files",
    )
    p.add_argument(
        "--topic",
        type=str,
        default="/openarm/joint_states",
        help="ROS2 topic to subscribe to for JointState messages",
    )
    return p.parse_args()


def main():
    args = parse_args()

    node = OpenArmStateReceiver(topic=args.topic)
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
