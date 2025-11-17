from threading import Event
import time

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller

from paradex.utils.keyboard_listener import listen_keyboard

rcc = remote_camera_controller("stream_main.py")

start_event = Event()
stop_event = Event()
end_event = Event()

listen_keyboard({"c":start_event, "q":stop_event, "e":end_event})

try:
    while not end_event.is_set():        
        if not start_event.is_set() and not end_event.is_set():
            time.sleep(0.01)
            continue
        if end_event.is_set():
            break
        
        rcc.start("stream", False, fps=10)
        print("Capture started. Press 'q' to stop.")
        while not stop_event.is_set():
            time.sleep(0.1)
        print("Stopping capture...")
        rcc.stop()
        start_event.clear()
        stop_event.clear()
        print("Capture stopped. Press 'c' to start again or 'e' to exit.")

finally:
    rcc.end()