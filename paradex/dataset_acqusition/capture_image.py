"""
Capture still images and the matching robot joint states on key press.

Press `c` to save:
  - one image from the remote camera system
  - current xArm qpos (arm_qpos.npy)
  - current hand qpos (hand_qpos.npy)
  - capture timestamp (timestamp.npy)

Press `q` to quit.
"""

import argparse
import os
import time
from threading import Event, Thread
from typing import Optional

import numpy as np
import chime

chime.theme("pokemon")

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.teleop.xsens.receiver import XSensReceiver
from paradex.utils.file_io import find_latest_index
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.path import shared_dir
from paradex.calibration.utils import save_current_camparam, save_current_C2R
from paradex.retargetor.state import HandStateExtractor
from paradex.retargetor.unimanual import Retargetor
from paradex.utils.system import network_info


class CaptureSession:
    """
    Single-frame capture with optional Xsens teleop control.
    """

    def __init__(
        self,
        arm: Optional[str] = None,
        hand: Optional[str] = None,
        tactile: bool = False,
        ip: bool = False,
        teleop: Optional[str] = None,
    ):
        # self.camera = remote_camera_controller(name="dataset_acquisition_image")
        self.arm = get_arm(arm) if arm else None
        self.hand = get_hand(hand_name=hand, tactile=tactile, ip=ip) if hand else None

        # Teleop (xsens)
        self.teleop_device = None
        self.retargetor = None
        self.state_extractor = None
        self.teleop_exit = Event()
        self.teleop_thread: Optional[Thread] = None

        if teleop == "xsens":
            self.teleop_device = XSensReceiver(**network_info["xsens"]["param"])
            self.retargetor = Retargetor(arm_name=arm, hand_name=hand)
            self.state_extractor = HandStateExtractor()
            self.teleop_thread = Thread(target=self._teleop_loop, daemon=True)
            self.teleop_thread.start()

    def capture_snapshot(self, save_path: str) -> None:
        """
        Capture a single image and save current arm/hand joint positions.

        Args:
            save_path: relative path (under shared_data) to the episode directory.
                       Example: capture/image/<name>/<index>
        """
        raw_dir = os.path.join(shared_dir, save_path, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # 1) trigger remote cameras for a single frame
        self.camera.start("image", False, save_path=os.path.join(save_path, "raw"))
        self.camera.stop()

        # 2) snapshot robot states
        ts = time.time()
        if self.arm is not None:
            arm_qpos = self.arm.get_data().get("qpos")
            np.save(os.path.join(raw_dir, "arm_qpos.npy"), arm_qpos)

        if self.hand is not None:
            hand_qpos = self.hand.get_data().get("qpos")
            np.save(os.path.join(raw_dir, "hand_qpos.npy"), hand_qpos)

        np.save(os.path.join(raw_dir, "timestamp.npy"), np.array([ts]))

        # 3) save calibration snapshots alongside capture
        save_current_camparam(os.path.join(shared_dir, save_path))
        save_current_C2R(os.path.join(shared_dir, save_path))

        print(f"Saved snapshot -> {save_path}")

    # ---------------------- Teleop loop (runs in background) ---------------------- #
    def _teleop_loop(self):
        if self.retargetor is None or self.teleop_device is None:
            return

        home_pose = self.arm.get_data()["position"] if self.arm is not None else np.eye(4)
        self.retargetor.start(home_pose)

        while not self.teleop_exit.is_set():
            data = self.teleop_device.get_data()
            if data["Right"] is None:
                time.sleep(0.005)
                continue

            state = self.state_extractor.get_state(data["Left"])

            if state == 0:
                wrist_pose, hand_action = self.retargetor.get_action(data)
                if self.hand is not None:
                    self.hand.move(hand_action)
                if self.arm is not None:
                    self.arm.move(wrist_pose.copy())
            elif state in (1, 2):
                self.retargetor.stop()

            if state == 3:
                # gesture-based exit, mirrors original behavior
                self.teleop_exit.set()

            time.sleep(0.01)

    def end(self):
        if self.arm is not None:
            self.arm.end()
        if self.hand is not None:
            self.hand.end()
        if self.teleop_device is not None:
            self.teleop_exit.set()
            if self.teleop_thread is not None:
                self.teleop_thread.join(timeout=1.0)
            self.teleop_device.end()
        # self.camera.end()


def main():
    parser = argparse.ArgumentParser(description="Capture single images with synchronized robot qpos.")
    parser.add_argument("--arm", type=str, default="xarm", help="Arm controller name (e.g., xarm)")
    parser.add_argument("--hand", type=str, default="allegro", help="Hand controller name (e.g., allegro)")
    parser.add_argument("--name", type=str, required=True, help="Session name")
    parser.add_argument("--tactile", action="store_true", help="Enable tactile hand (Inspire)")
    parser.add_argument("--ip", action="store_true", help="Use IP connection for hand (Inspire)")
    parser.add_argument("--teleop", action="store_true", help="Enable Xsens teleoperation")
    args = parser.parse_args()

    teleop_mode = "xsens" if args.teleop else None
    cs = CaptureSession(arm=args.arm, hand=args.hand, tactile=args.tactile, ip=args.ip, teleop=teleop_mode)

    capture_event = Event()
    exit_event = Event()
    listen_keyboard({"c": capture_event, "q": exit_event})
    print("Press 'c' to capture one frame; 'q' to quit.")

    base_dir = os.path.join("capture", "image", args.name)
    last_idx = int(find_latest_index(os.path.join(shared_dir, base_dir)))

    try:
        while not exit_event.is_set():
            if not capture_event.wait(timeout=0.01):
                continue
            capture_event.clear()

            last_idx += 1
            save_path = os.path.join(base_dir, str(last_idx))
            cs.capture_snapshot(save_path)

    finally:
        cs.end()


if __name__ == "__main__":
    main()
