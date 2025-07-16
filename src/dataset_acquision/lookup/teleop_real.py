
import numpy as np
import argparse
import os

from paradex.io.robot_controller import XArmController, AllegroController, InspireController
from paradex.io.teleop import XSensReceiver, OculusReceiver
from paradex.io.contact.receiver import SerialReader
from paradex.retargetor import Unimanual_Retargetor, HandStateExtractor
from paradex.geometry.coordinate import DEVICE2WRIST

from threading import Event
from paradex.utils.keyboard_listener import listen_keyboard
import time

from paradex.robot import RobotWrapper
from paradex.utils.file_io import rsc_path

import chime

parser = argparse.ArgumentParser()

parser.add_argument('--device', choices=['xsens', 'occulus'])
parser.add_argument('--arm', type=str)
parser.add_argument('--hand', type=str)
parser.add_argument('--save_path', type=str)

args = parser.parse_args()

def initialize_teleoperation(save_path):
    controller = {}
    
    if args.arm == "xarm":
        controller["arm"] = XArmController(save_path)

    if args.hand == "allegro":
        controller["hand"] = AllegroController(save_path)
        if save_path != None:
            controller["contact"] = SerialReader(save_path)
    
    elif args.hand == "inspire":
        controller["hand"] = InspireController(save_path)
    
    if args.device == "xsens":
        controller["teleop"] = XSensReceiver()
    if args.device == "occulus":
        controller["teleop"] = OculusReceiver()

    return controller

def main():    
    save_path = args.save_path
    sensors = initialize_teleoperation(save_path)
    
    state_extractor = HandStateExtractor()
    home_pose = np.eye(4) # Don't care
    if "arm" in sensors:
        home_pose = sensors["arm"].get_position().copy()
    # home_pose = np.array([
    #                     [0, 1 ,0, 0.3],
    #                     [0, 0, 1, -0.2],
    #                     [1, 0, 0, 0.2],
    #                     [0, 0, 0, 1]]
    #                     )
    
    retargetor = Unimanual_Retargetor(args.arm, args.hand, home_pose)
    if "arm" in sensors:
        sensors["arm"].home_robot(home_pose)
        home_start_time = time.time()
        while not sensors["arm"].is_ready():
            if time.time() - home_start_time > 0.3:
                chime.warning()
                home_start_time = time.time()
            time.sleep(0.0008)
        chime.success()
    
    while True:
        data = sensors["teleop"].get_data()
        if data["Right"] is None:
            continue
        state = state_extractor.get_state(data['Left'])
        if state == 1:
            retargetor.pause()
            continue
        
        if state == 2:
            break
        
        wrist_pose, hand_action = retargetor.get_action(data)
      

        if args.hand is not None:
            sensors["hand"].set_target_action(hand_action)
        
        if args.arm is not None:
            sensors["arm"].set_action(wrist_pose.copy())
            
        time.sleep(0.03)
        
    for key in sensors.keys():
        sensors[key].quit()
        
    print("Program terminated.")

if __name__ == "__main__":
    main()