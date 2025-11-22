from threading import Event
import time

from paradex.io.capture_pc.ssh import run_script
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.utils.keyboard_listener import listen_keyboard

start_event = Event()
exit_event = Event()
stop_event = Event()

listen_keyboard({"c": start_event, "q": exit_event, "s": stop_event})

dc = DataCollector()
dc.start()

while not exit_event.is_set():
    if start_event.is_set() and not stop_event.is_set():
        print("Streaming started...")
        data_dict = dc.get_data()
        for pc_id, data in data_dict.items():
            print(f"PC ID: {pc_id}, Data: {data}")
    time.sleep(0.1)

    if stop_event.is_set():
        start_event.clear()
        stop_event.clear()
        print("Streaming stopped.")
        continue