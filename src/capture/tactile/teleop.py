
import numpy as np
from scipy.spatial.transform import Rotation

from paradex.io.robot_controller import XArmController, AllegroController, InspireController
from paradex.io.xsens.receiver import XSensReceiver
from paradex.io.contact.receiver import SerialReader
from paradex.io.camera.camera_loader import CameraManager
from paradex.retargetor.unimanual import Retargetor

import time
import threading
import os
import chime
import transforms3d as t3d
from paradex.utils.keyboard_listener import listen_keyboard
import matplotlib.pyplot as plt
from collections import deque


hand_name = "allegro"
arm_name = None# "xarm"

home_wrist_pose = np.array([[0, 1 ,0, 0.3],[0, 0, 1, -0.2],[1, 0, 0, 0.2],[0, 0, 0, 1]])

def load_homepose(hand_name):
    if hand_name == "allegro":
        return  np.load("data/home_pose/allegro_hand_joint_angle.npy")
    elif hand_name == "inspire":
        return np.zeros(6)+500
    

def homo2cart(h):
    if h.shape == (4, 4):
        t = h[:3, 3]
        R = h[:3, :3]

        axis, angle = t3d.axangles.mat2axangle(R)
        axis_angle = axis * angle
    else:
        raise ValueError("Invalid input shape.")
    return np.concatenate([t, axis_angle])

def listen_for_exit(stop_event):
    """Listens for 'q' key input to safely exit all processes."""
    while not stop_event.is_set():
        user_input = input()
        if user_input.lower() == "q":
            print("\n[INFO] Exiting program...")
            stop_event.set()  # Set the exit flag
            break

def initialize_teleoperation(save_path):
    controller = {}        

    if arm_name == "xarm":
        controller["arm"] = XArmController(save_path)

    if hand_name == "allegro":
        controller["hand"] = AllegroController(save_path)
        controller["contact"] = SerialReader(save_path)
        
    elif hand_name == "inspire":
        controller["hand"] = InspireController(save_path)
    
    
    controller["xsens"] = XSensReceiver()

    return controller

# 최근 1000개 저장용 deque
contact_buffer = deque(maxlen=100)

# 시각화 준비
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_ylim(0, 32767)
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Contact Sensor Value")
ax.set_title("Real-Time Contact Sensor (Last 1000 Samples)")
ax.grid()

time_axis = np.linspace(-100 / 30.0, 0, 100)  # 30Hz 기준
lines = []
for i in range(15):
    line, = ax.plot(time_axis, np.zeros(100), label=f"Sensor {i}")
    lines.append(line)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.ion()
plt.show()

def main():    
    save_path ="tmotactile" 
    print (f"save_path: {save_path}")
    sensors = initialize_teleoperation(save_path)
    
    stop_event = threading.Event()
    listen_keyboard({"q":stop_event})

    retargetor = Retargetor(arm_name, hand_name, home_wrist_pose)

    homepose_cnt = 0
    home_hand_pose = load_homepose(hand_name)
    asdf = 0
    
    while True:
        asdf += 1
        if stop_event.is_set():
            break

        if hand_name is not None:
            sensors["hand"].set_homepose(home_hand_pose)
            sensors["hand"].home_robot()
            
        data = sensors["xsens"].get_data()
        if data['Right'] == None:
            continue
        
        arm_action, hand_action = retargetor.get_action(data)

        sensors["hand"].set_target_action(hand_action)
        tactile_sensor_value = sensors["contact"].get_data()
        if tactile_sensor_value.ndim == 1:
            contact_buffer.append(tactile_sensor_value.copy())
        print(tactile_sensor_value)
        # 시각화 업데이트
        if len(contact_buffer) == 100:
            buf = np.array(contact_buffer)  # shape: (1000, 15)
            for i in range(15):
                lines[i].set_ydata(buf[:, i])

            fig.canvas.draw()
            fig.canvas.flush_events()

        time.sleep(1.0 / 30.0)
    for key in sensors.keys():
        sensors[key].quit()

    print("Program terminated.")
    exit(0)

if __name__ == "__main__":
    main()