from threading import Event
import time

from paradex.io.capture_pc.data_sender import DataPublisher
from paradex.io.capture_pc.command_sender import CommandReceiver

start_event = Event()
exit_event = Event()
stop_event = Event()    

dp = DataPublisher()
cr = CommandReceiver({"start": start_event, "exit": exit_event, "stop": stop_event})

start_time = time.time()    
while not exit_event.is_set():
    if start_event.is_set() and not stop_event.is_set():
        dp.send_data({"value": time.time() - start_time})
    time.sleep(0.1)

    if stop_event.is_set():
        print("Stopped")
        start_event.clear()
        stop_event.clear()
        continue

