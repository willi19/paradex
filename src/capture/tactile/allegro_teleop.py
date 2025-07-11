
import numpy as np

from paradex.io.robot_controller import AllegroController
from paradex.io.xsens.receiver import XSensReceiver
from paradex.io.contact.receiver import SerialReader
from paradex.retargetor.unimanual import Retargetor

import time
import threading
import transforms3d as t3d
from paradex.utils.keyboard_listener import listen_keyboard
import matplotlib.pyplot as plt
from collections import deque


hand_name = "allegro"

def initialize_teleoperation(save_path):
    controller = {}        

    if hand_name == "allegro":
        controller["hand"] = AllegroController(save_path)
        controller["contact"] = SerialReader(save_path)
        
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

    retargetor = Retargetor(None, hand_name, np.eye(4))

    home_hand_pose = np.zeros(16)
    while not stop_event.is_set():
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