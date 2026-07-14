import os
import time
import glob
import subprocess
import numpy as np
import chime
chime.theme('pokemon')

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.tactile import HumanTactileRecorder
from paradex.utils.path import shared_dir
from paradex.retargetor.state import HandStateExtractor
from paradex.retargetor.unimanual import Retargetor
from paradex.calibration.utils import save_current_camparam, save_current_C2R
from paradex.utils.system import network_info


def _normalize_optional_name(name):
    if name is not None and isinstance(name, str) and name.strip().lower() in ("", "none", "null"):
        return None
    return name


class CaptureSession():
    def __init__(
        self,
        camera=False,
        arm=None,
        hand=None,
        hand_left=None,
        hand_right=None,
        teleop=None,
        tactile=False,
        ip=False,
        hand_side="right",
        events=None,
        realsense=False,
        arm_kwargs=None,
        camera_pc_list=None,
        hand_scale=1.0,
        human_tactile=False,
        human_tactile_port="/dev/ttyUSB0",
        human_tactile_baud_rate=115200,
        human_tactile_reset_wait=2.0,
        human_tactile_plot_realtime=False,
        human_tactile_plot_refresh_interval=0.02,
        human_tactile_plot_max_samples=200,
    ):
        arm = _normalize_optional_name(arm)
        hand = _normalize_optional_name(hand)
        hand_left = _normalize_optional_name(hand_left)
        hand_right = _normalize_optional_name(hand_right)

        if realsense:
            from paradex.io.camera_system.realsense_controller import realsense_controller
            self.realsense = realsense_controller()
        else:
            self.realsense = None
        
        if arm is None and hand is None and hand_left is None and hand_right is None and teleop is not None:
            raise ValueError("Teleop device requires at least one of arm or hand to be specified.")
        
        if camera:
            self.camera = remote_camera_controller(name="dataset_acquisition", pc_list=camera_pc_list)
            self.sync_generator = UTGE900(**network_info["signal_generator"]["param"])
            # if arm is not None or hand is not None:
            self.timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])
            # else:
            #     self.timestamp_monitor = None
        else:
            self.camera = None
            self.timestamp_monitor = None
            self.sync_generator = None

     

        if hand_side not in ["right", "left", "bimanual"]:
            raise ValueError("Not supported hand side")

        self.events = events

        if hand_side == "right":
            self.hand_side = "Right"
            self.hand_side_opposite = "Left"
        elif hand_side == "left":
            self.hand_side = "Left"
            self.hand_side_opposite = "Right"
        else:
            self.hand_side = "Bimanual"

        self.arm = None
        self.arm_left = None
        self.arm_right = None
        self.arm_name = None
        if arm is not None:
            arm_kwargs = {} if arm_kwargs is None else dict(arm_kwargs)
            self.arm_name = arm
            if self.hand_side == "Bimanual":
                left_kwargs = arm_kwargs.pop("left", arm_kwargs.copy())
                right_kwargs = arm_kwargs.pop("right", arm_kwargs.copy())
                left_kwargs.setdefault("namespace", "left")
                right_kwargs.setdefault("namespace", "right")
                self.arm_right = get_arm(arm, **right_kwargs)
                self.arm_left = get_arm(arm, **left_kwargs)
            else:
                self.arm = get_arm(arm, **arm_kwargs)
        
        self.hand = None
        self.hand_left = None
        self.hand_right = None
        self.hand_name = None
        self.hand_name_left = None
        self.hand_name_right = None

        if self.hand_side == "Bimanual":
            left_name = hand_left if hand_left is not None else hand
            right_name = hand_right if hand_right is not None else hand
            if left_name is not None:
                self.hand_name_left = left_name
                self.hand_left = get_hand(hand_name=left_name, tactile=tactile, ip=ip, hand_side="left")
            if right_name is not None:
                self.hand_name_right = right_name
                self.hand_right = get_hand(hand_name=right_name, tactile=tactile, ip=ip, hand_side="right")
        else:
            if hand is not None:
                self.hand_name = hand
                self.hand = get_hand(
                    hand_name=hand,
                    tactile=tactile,
                    ip=ip,
                    hand_side=self.hand_side.lower(),
                )

        

            
        if teleop is not None:
            if teleop == "xsens":
                # if arm == "openarm":
                from paradex.io.teleop.xsens.receiver import XSensReceiver
                self.teleop_device = XSensReceiver(**network_info["xsens"]["param"])
                
            # elif teleop == "occulus":
            #     from paradex.io.teleop.oculus.receiver import OculusReceiver
            #     self.teleop_device = OculusReceiver()
            if arm != 'openarm':
                self.retargetor = Retargetor(
                    arm_name=arm,
                    hand_name=hand,
                    hand_side=self.hand_side,
                    hand_name_left=self.hand_name_left,
                    hand_name_right=self.hand_name_right,
                    hand_scale=hand_scale,
                )
                self.state_extractor = HandStateExtractor()

        else:
            self.teleop_device = None

        self.human_tactile = None
        if human_tactile:
            self.human_tactile = HumanTactileRecorder(
                port=human_tactile_port,
                baud_rate=human_tactile_baud_rate,
                reset_wait=human_tactile_reset_wait,
                plot_realtime=human_tactile_plot_realtime,
                plot_refresh_interval=human_tactile_plot_refresh_interval,
                plot_max_samples=human_tactile_plot_max_samples,
            )
            self.human_tactile.connect()
            
        self.save_path = None
        self._camera_capture_started = False
        self._timestamp_monitor_started = False
        self._sync_generator_started = False
            
    def start(self, save_path): # Start recording on all sensors
        print("Starting new capture session, saving to:", save_path)
        self.save_path = save_path
        os.makedirs(os.path.join(shared_dir, save_path, "raw"), exist_ok=True)

        if self.human_tactile is not None:
            self.human_tactile.start(os.path.join(shared_dir, save_path, "raw", "human_tactile"))
        
        if self.arm is not None:
            self.arm.start(os.path.join(shared_dir, save_path, "raw", "arm"))
        if self.arm_left is not None:
            self.arm_left.start(os.path.join(shared_dir, save_path, "raw", "arm_left"))
        if self.arm_right is not None:
            self.arm_right.start(os.path.join(shared_dir, save_path, "raw", "arm_right"))
        
        if self.hand is not None:
            self.hand.start(os.path.join(shared_dir, save_path, "raw", "hand"))
        if self.hand_left is not None:
            self.hand_left.start(os.path.join(shared_dir, save_path, "raw", "hand_left"))
        if self.hand_right is not None:
            self.hand_right.start(os.path.join(shared_dir, save_path, "raw", "hand_right"))
            
        if self.teleop_device is not None:
            self.teleop_device.start(os.path.join(shared_dir, save_path, "raw", "teleop"))
            self.state_hist = []
            self.state_time = []

        if self.camera is not None:
            # self.sync_generator.start(fps=30)
            self.camera.start("video", True, os.path.join(save_path, "raw"))
            self._camera_capture_started = True
            if self.timestamp_monitor is not None:
                self.timestamp_monitor.start(os.path.join(shared_dir, save_path, "raw", "timestamps"))
                self._timestamp_monitor_started = True
            self.sync_generator.start(fps=30)
            self._sync_generator_started = True
            try:
                self.camera.validate(timeout=5.0)
            except Exception:
                # Stop remote pipelines while pulses are still present so
                # aravissrc/avimux can finalize without trigger starvation.
                try:
                    self.camera.stop()
                finally:
                    self._camera_capture_started = False
                    self.sync_generator.stop()
                    self._sync_generator_started = False
                raise
        
        if self.realsense is not None:
            self.realsense.start(
                save_path=os.path.join(shared_dir, save_path, "depth_cam"),
                fps=30,
                use_depth=True,
            )
        
        
    def stop(self):
        if self.arm is not None:
            self.arm.stop()
        if self.arm_left is not None:
            self.arm_left.stop()
        if self.arm_right is not None:
            self.arm_right.stop()
            
        if self.hand is not None:
            self.hand.stop()
        if self.hand_left is not None:
            self.hand_left.stop()
        if self.hand_right is not None:
            self.hand_right.stop()

        if self.human_tactile is not None:
            self.human_tactile.stop()
                
        if self.teleop_device is not None:
            self.teleop_device.stop()
            os.makedirs(os.path.join(shared_dir, self.save_path, "raw", "state"), exist_ok=True)
            np.save(os.path.join(shared_dir, self.save_path, "raw", "state", "state_hist.npy"), np.array(self.state_hist))
            np.save(os.path.join(shared_dir, self.save_path, "raw", "state", "state_time.npy"), np.array(self.state_time))

        if self.camera is not None:
            try:
                if self._camera_capture_started:
                    print("Stopping camera and saving calibration data...")
                    # ParaOffice keeps hardware trigger pulses alive until
                    # camera branches are closed. Stopping UTG first starves
                    # aravissrc during AVI EOS/finalization.
                    self.camera.stop()
                    self._camera_capture_started = False
                    print("Camera stopped.")
                    if self._timestamp_monitor_started and self.timestamp_monitor is not None:
                        self.timestamp_monitor.stop()
                        self._timestamp_monitor_started = False

                    save_current_camparam(os.path.join(shared_dir, self.save_path))
                    if self.arm is not None or self.arm_left is not None or self.arm_right is not None:
                        if self.arm_name == "xarm":
                            save_current_C2R(os.path.join(shared_dir, self.save_path))
                        elif self.arm_name == "openarm":
                            save_current_C2R(os.path.join(shared_dir, self.save_path), arm="openarm")
            finally:
                if self._sync_generator_started:
                    self.sync_generator.stop()
                    self._sync_generator_started = False

        if self.realsense is not None:
            self.realsense.stop()
        self.save_path = None

    def end(self):
        if self.arm is not None:
            self.arm.end()
        if self.arm_left is not None:
            self.arm_left.end()
        if self.arm_right is not None:
            self.arm_right.end()
        
        if self.hand is not None:
            self.hand.end()
        if self.hand_left is not None:
            self.hand_left.end()
        if self.hand_right is not None:
            self.hand_right.end()
        if self.teleop_device is not None:
            self.teleop_device.end()
        
        if self.camera is not None:
            self.camera.end()
            if self.timestamp_monitor is not None:
                self.timestamp_monitor.end()
            self.sync_generator.end()
        if self.realsense is not None:
            self.realsense.end()
        if self.human_tactile is not None:
            self.human_tactile.close()
    
    def teleop(self, session_events=None, state_policy="gesture_control", loop_callback=None):
        if self.teleop_device is None:
            raise ValueError("No teleop device initialized.")
        if state_policy not in ["gesture_control", "keyboard_control"]:
            raise ValueError(f"Unknown state_policy: {state_policy}")

        if session_events is None:
            session_events = self.events

        chime.warning(sync=True)
        exit_counter = 0
        stop_counter = 0

        if self.hand_side == "Bimanual":
            if self.arm_left is not None and self.arm_right is not None:
                home_pose = {
                    "Left": self.arm_left.get_data()["position"],
                    "Right": self.arm_right.get_data()["position"],
                }
            else:
                home_pose = {"Left": np.eye(4), "Right": np.eye(4)}
        else:
            home_pose = self.arm.get_data()["position"] if self.arm is not None else np.eye(4)

        self.retargetor.start(home_pose)

        while True:
            if session_events is not None and session_events["exit"].is_set():
                chime.success(sync=True)
                return "exit"

            data = self.teleop_device.get_data()

            
            if self.hand_side != "Bimanual":
                if data[self.hand_side] is None:
                    print("No data from teleop device...")
                    if session_events is not None:
                        if self.save_path is None and session_events["save"].is_set():
                            return "start"
                        if self.save_path is not None and session_events["stop"].is_set():
                            chime.info(sync=True)
                            return "stop"
                    time.sleep(0.01)
                    continue

                state = self.state_extractor.get_state(data[self.hand_side_opposite])

                if self.save_path is not None:
                    self.state_hist.append(state)
                    self.state_time.append(time.time())

                if loop_callback is not None:
                    try:
                        loop_callback(self)
                    except Exception as exc:
                        print(f"teleop loop_callback failed: {exc}")
                        loop_callback = None
                    
                if state == 0:
                    wrist_pose, hand_action = self.retargetor.get_action(data)
                    if self.hand is not None:
                        self.hand.move(hand_action)

                    if self.arm is not None:
                        self.arm.move(wrist_pose.copy())

                if state == 1:   
                    self.retargetor.stop()
                
                if state == 2:
                    self.retargetor.stop()
                    if state_policy == "gesture_control":
                        stop_counter += 1
                
                elif state_policy == "gesture_control":
                    stop_counter = 0
                    
                if state == 3:
                    if state_policy == "gesture_control":
                        exit_counter += 1
                
                elif state_policy == "gesture_control":
                    exit_counter = 0

                if state_policy == "gesture_control":
                    if exit_counter > 90:
                        chime.success(sync=True)
                        return "exit"
                
                    if stop_counter > 90:
                        chime.info(sync=True)
                        return "stop"

                if session_events is not None:
                    if self.save_path is None and session_events["save"].is_set():
                        return "start"
                    if self.save_path is not None and session_events["stop"].is_set():
                        chime.info(sync=True)
                        return "stop"
            
            else:
                if data["Left"] is None or data["Right"] is None:
                    print("No data from teleop device...")
                    if session_events is not None:
                        if self.save_path is None and session_events["save"].is_set():
                            return "start"
                        if self.save_path is not None and session_events["stop"].is_set():
                            chime.info(sync=True)
                            return "stop"
                    time.sleep(0.01)
                    continue

                wrist_pose_left, wrist_pose_right, hand_action_left, hand_action_right = self.retargetor.get_action(data)

                if self.arm_left is not None:
                    self.arm_left.move(wrist_pose_left.copy())
                if self.arm_right is not None:
                    self.arm_right.move(wrist_pose_right.copy())
                if self.hand_left is not None and hand_action_left is not None:
                    self.hand_left.move(hand_action_left)
                if self.hand_right is not None and hand_action_right is not None:

                    self.hand_right.move(hand_action_right)

                if session_events is not None:
                    if self.save_path is None and session_events["save"].is_set():
                        return "start"
                    if self.save_path is not None and session_events["stop"].is_set():
                        chime.info(sync=True)
                        return "stop"

            # else:
            #     if data is None:
            #         continue
            #     if self.hand_left is not None:
            #         wrist_pose, hand_action_left, hand_action_right = self.retargetor.get_action(data)
                    
                    
            #         self.hand_left.move(hand_action_left)
            #     if self.events["save"].is_set():
            #         return "saving"
            #     if self.events["stop"].is_set():
            #         return "stop"
            #     if self.events["exit"].is_set():
            #         return "exit"
            time.sleep(0.01)
        
    
    def move(self, action_dict):
        if "arm" in action_dict and self.arm is not None:
            self.arm.move(action_dict["arm"])
        if "hand" in action_dict and self.hand is not None:
            self.hand.move(action_dict["hand"])
