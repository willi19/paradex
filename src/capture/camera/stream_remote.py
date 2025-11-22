from threading import Event
import time

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.capture_pc.ssh import run_script
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.utils.keyboard_listener import listen_keyboard


run_script("python src/capture/stream_client.py")

rcc = remote_camera_controller("stream_main.py")
dc = DataCollector()

start_event = Event()
exit_event = Event()

listen_keyboard({"q":exit_event})

rcc.start("stream", False, fps=10)

while not exit_event.is_set():        
    time.sleep(0.1)

print("Stopping capture...")
rcc.stop()
rcc.end()     


 