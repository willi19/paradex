import os
import time
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
    def __init__(self, camera=False, arm=None, hand=None, teleop=None, tactile = False, ip = False, hand_side = "right", events = None):
        if arm is None and hand is None and teleop is not None:
            raise ValueError("Teleop device requires at least one of arm or hand to be specified.")
        
        if camera:
            self.camera = remote_camera_controller(name="dataset_acquisition")
            self.sync_generator = UTGE900(**network_info["signal_generator"]["param"])
            if arm is not None or hand is not None:
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
        
        if hand is not None:
            if self.hand_side != "Bimanual":
                self.hand = get_hand(hand_name = hand, tactile = tactile, ip = ip)
            else:
                self.hand_left = get_hand(hand_name = hand, tactile = tactile, ip = ip)

                # self.hand_left = get_hand(hand_name = hand, tactile = tactile, ip = ip)
                # self.hand_right = get_hand(hand_name = hand, tactile = tactile, ip = ip)
        else:
            self.hand = None
        

            
        if teleop is not None:
            if teleop == "xsens":
                # if arm == "openarm":
                from paradex.io.teleop.xsens.receiver import XSensReceiver
                self.teleop_device = XSensReceiver(**network_info["xsens"]["param"])
                
            # elif teleop == "occulus":
            #     from paradex.io.teleop.oculus.receiver import OculusReceiver
            #     self.teleop_device = OculusReceiver()
            if arm == 'openarm':
                self.retargetor = Retargetor(arm_name=None, hand_name=hand, hand_side = self.hand_side)
            else:
                self.retargetor = Retargetor(arm_name=arm, hand_name=hand, hand_side = self.hand_side)
                self.state_extractor = HandStateExtractor()

        else:
            self.teleop_device = None
            
        self.save_path = None
            
    def start(self, save_path): # Start recording on all sensors
        print("Starting new capture session, saving to:", save_path)
        self.save_path = save_path
        os.makedirs(os.path.join(shared_dir, save_path, "raw"), exist_ok=True)
        
        if self.arm is not None:
            self.arm.start(os.path.join(shared_dir, save_path, "raw", "arm"))
        
        if self.hand_side != "Bimanual":
            if self.hand is not None:
                self.hand.start(os.path.join(shared_dir, save_path, "raw", "hand"))
                
        else:
            if self.hand_left is not None:
                self.hand_left.start(os.path.join(shared_dir, save_path, "raw", "hand_left"))
            # if self.hand_right is not None:
            #     self.hand_right.start(os.path.join(shared_dir, save_path, "raw", "hand_right"))
            
        if self.teleop_device is not None:
            self.teleop_device.start(os.path.join(shared_dir, save_path, "raw", "teleop"))
            self.state_hist = []
            self.state_time = []

        if self.camera is not None:
            self.camera.start("video", True, os.path.join(save_path, "raw"))
            if self.timestamp_monitor is not None:
                self.timestamp_monitor.start(os.path.join(shared_dir, save_path, "raw", "timestamps"))
            self.sync_generator.start(fps=30)
        
    def stop(self):
        if self.arm is not None:
            self.arm.stop()
            
        if self.hand_side is not "Bimanual":
            if self.hand is not None:
                self.hand.stop()
                
        else:
            if self.hand_left is not None:
                self.hand_left.stop()
            # if self.hand_right is not None:
            #     self.hand_right.stop()
                
        if self.teleop_device is not None:
            self.teleop_device.stop()
            os.makedirs(os.path.join(shared_dir, self.save_path, "raw", "state"), exist_ok=True)
            np.save(os.path.join(shared_dir, self.save_path, "raw", "state", "state_hist.npy"), np.array(self.state_hist))
            np.save(os.path.join(shared_dir, self.save_path, "raw", "state", "state_time.npy"), np.array(self.state_time))

        if self.camera is not None:
            print("Stopping camera and saving calibration data...")
            self.camera.stop()
            print("Camera stopped.")
            if self.timestamp_monitor is not None:
                self.timestamp_monitor.stop()
            self.sync_generator.stop()
        
            save_current_camparam(os.path.join(shared_dir, self.save_path))
            save_current_C2R(os.path.join(shared_dir, self.save_path))
        
        self.save_path = None

    def end(self):
        if self.arm is not None:
            self.arm.end()
        
        if self.hand_side != "Bimanual":
            if self.hand is not None:
                self.hand.end()
        else:
            if self.hand_left is not None:
                self.hand_left.end()
            # if self.hand_right is not None:
            #     self.hand_right.end()
        if self.teleop_device is not None:
            self.teleop_device.end()
        
        if self.camera is not None:
            self.camera.end()
            if self.timestamp_monitor is not None:
                self.timestamp_monitor.end()
            self.sync_generator.end()
    
    def teleop(self, stop = None):
        if self.teleop_device is None:
            raise ValueError("No teleop device initialized.")

        chime.warning(sync=True)
        exit_counter = 0
        stop_counter = 0

        home_pose = self.arm.get_data()["position"] if self.arm is not None else np.eye(4)
        
        self.retargetor.start(home_pose)

        while True:
            data = self.teleop_device.get_data()

            
            if self.hand_side != "Bimanual":
                if data[self.hand_side] is None:
                    continue
                state = self.state_extractor.get_state(data[self.hand_side_opposite])
                if self.save_path is not None:
                    self.state_hist.append(state)
                    self.state_time.append(time.time())
                    
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
                if data is None:
                    continue
                if self.hand_left is not None:
                    wrist_pose, hand_action_left, hand_action_right = self.retargetor.get_action(data)
                    
                    
                    self.hand_left.move(hand_action_left)
                if self.events["save"].is_set():
                    return "saving"
                if self.events["stop"].is_set():
                    return "stop"
                if self.events["exit"].is_set():
                    return "exit"
            time.sleep(0.01)
        
    
    def move(self, action_dict):
        if "arm" in action_dict:
            self.arm.move(action_dict["arm"])
        if "hand" in action_dict:
            self.hand.move(action_dict["hand"])()
