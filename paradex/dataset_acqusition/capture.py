import os
import time
import json
import numpy as np
import chime
chime.theme('pokemon')

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.io.robot_controller import get_arm, get_hand
from paradex.utils.path import shared_dir
from paradex.retargetor.state import HandStateExtractor
from paradex.retargetor.unimanual import Retargetor
from paradex.calibration.utils import save_current_camparam, save_current_C2R
from paradex.utils.system import network_info

class CaptureSession():
    def __init__(self, camera=False, arm=None, hand=None, teleop=None, hand_ip=False,
                 use_sync_gen=True, camera_fps=30, use_timestamp_monitor=True,
                 translation_scale=1.0, camera_mode="full"):
        if arm is None and hand is None and teleop is not None:
            raise ValueError("Teleop device requires at least one of arm or hand to be specified.")

        self.use_sync_gen = use_sync_gen if camera else False
        self.camera_fps = camera_fps
        # "full" records {serial}.avi AND streams frames to shared memory so
        # live consumers (e.g. object-6d-tracking via CameraReader) can read the
        # same frames while recording. "video" = record only (no streaming).
        self.camera_mode = camera_mode

        if camera:
            self.camera = remote_camera_controller(name="dataset_acquisition")
            if use_sync_gen:
                self.sync_generator = UTGE900(**network_info["signal_generator"]["param"])
            else:
                self.sync_generator = None
            if use_timestamp_monitor and (arm is not None or hand is not None):
                self.timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])
            else:
                self.timestamp_monitor = None
        else:
            self.camera = None
            self.timestamp_monitor = None
            self.sync_generator = None
        
        if arm is not None:
            self.arm = get_arm(arm)
        else:
            self.arm = None
        
        if hand is not None:
            self.hand = get_hand(hand, ip=hand_ip)
        else:
            self.hand = None
            
        if teleop is not None:
            if teleop == "xsens":
                from paradex.io.teleop.xsens.receiver import XSensReceiver
                self.teleop_device = XSensReceiver(**network_info["xsens"]["param"])
            
            # elif teleop == "occulus":
            #     from paradex.io.teleop.oculus.receiver import OculusReceiver
            #     self.teleop_device = OculusReceiver()
            self.retargetor = Retargetor(arm_name=arm, hand_name=hand,
                                         translation_scale=translation_scale)
            self.state_extractor = HandStateExtractor()
            
        else:
            self.teleop_device = None
            
        self.save_path = None
            
    def start(self, save_path): # Start recording on all sensors
        self.save_path = save_path
        os.makedirs(os.path.join(shared_dir, save_path, "raw"), exist_ok=True)
        
        if self.arm is not None:
            self.arm.start(os.path.join(shared_dir, save_path, "raw", "arm"))
            
        if self.hand is not None:
            self.hand.start(os.path.join(shared_dir, save_path, "raw", "hand"))
            
        if self.teleop_device is not None:
            self.teleop_device.start(os.path.join(shared_dir, save_path, "raw", "teleop"))
            self.state_hist = []
            self.state_time = []

        if self.camera is not None:
            self.camera.start(self.camera_mode, self.use_sync_gen, os.path.join(save_path, "raw"), fps=self.camera_fps)
            if self.timestamp_monitor is not None:
                self.timestamp_monitor.start(os.path.join(shared_dir, save_path, "raw", "timestamps"))
            if self.sync_generator is not None:
                self.sync_generator.start(fps=self.camera_fps)
            # Start-time anchor: main-PC wall clock at the moment cameras were
            # commanded to start. Video frame i (blank-filled => uniform) maps
            # to wall time ~ camera_start_time + i / fps. Residual (network +
            # FPGA startup, ~const) is tuned out via projection_check --offset.
            camera_start_time = time.time()
            with open(os.path.join(shared_dir, save_path, "camera_meta.json"), "w") as f:
                json.dump({"start_time": camera_start_time,
                           "fps": self.camera_fps,
                           "sync_gen": bool(self.sync_generator is not None)}, f)
        
    def stop(self):
        if self.arm is not None:
            self.arm.stop()
        if self.hand is not None:
            self.hand.stop()
            
        if self.teleop_device is not None:
            self.teleop_device.stop()
            os.makedirs(os.path.join(shared_dir, self.save_path, "raw", "state"), exist_ok=True)
            np.save(os.path.join(shared_dir, self.save_path, "raw", "state", "state_hist.npy"), np.array(self.state_hist))
            np.save(os.path.join(shared_dir, self.save_path, "raw", "state", "state_time.npy"), np.array(self.state_time))

        if self.camera is not None:
            self.camera.stop()
            if self.timestamp_monitor is not None:
                self.timestamp_monitor.stop()
            if self.sync_generator is not None:
                self.sync_generator.stop()

            save_current_camparam(os.path.join(shared_dir, self.save_path))
            save_current_C2R(os.path.join(shared_dir, self.save_path))
        
        self.save_path = None

    def end(self):
        if self.arm is not None:
            self.arm.end()
        if self.hand is not None:
            self.hand.end()
        if self.teleop_device is not None:
            self.teleop_device.end()
        
        if self.camera is not None:
            self.camera.end()
            if self.timestamp_monitor is not None:
                self.timestamp_monitor.end()
            if self.sync_generator is not None:
                self.sync_generator.end()
    
    def teleop(self, stop_event=None, exit_event=None, use_gesture_exit=True):
        if self.teleop_device is None:
            raise ValueError("No teleop device initialized.")

        chime.warning(sync=True)
        exit_counter = 0
        stop_counter = 0

        home_pose = self.arm.get_data()["position"] if self.arm is not None else np.eye(4)

        self.retargetor.start(home_pose)

        while True:
            if stop_event is not None and stop_event.is_set():
                stop_event.clear()
                chime.info(sync=True)
                return "stop"
            if exit_event is not None and exit_event.is_set():
                exit_event.clear()
                chime.success(sync=True)
                return "exit"

            data = self.teleop_device.get_data()
            if data["Right"] is None:
                continue
            state = self.state_extractor.get_state(data['Left'])
            if self.save_path is not None:
                self.state_hist.append(state)
                self.state_time.append(time.time())

            if state == 0:
                wrist_pose, hand_action = self.retargetor.get_action(data)
                if self.hand is not None:
                    self.hand.move(hand_action)

                if self.arm is not None:
                    self.arm.move(wrist_pose.copy())

            if use_gesture_exit:
                if state == 1:
                    self.retargetor.stop()

                if state == 2:
                    self.retargetor.stop()
                    stop_counter += 1
                else:
                    stop_counter = 0

                if state == 3:
                    exit_counter += 1
                else:
                    exit_counter = 0

                if exit_counter > 90:
                    chime.success(sync=True)
                    return "exit"
                if stop_counter > 90:
                    chime.info(sync=True)
                    return "stop"
            else:
                if state != 0:
                    self.retargetor.stop()

            time.sleep(0.01)
        
    
    def move(self, action_dict):
        if "arm" in action_dict:
            self.arm.move(action_dict["arm"])
        if "hand" in action_dict:
            self.hand.move(action_dict["hand"])
