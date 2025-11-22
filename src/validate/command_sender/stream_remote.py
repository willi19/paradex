from threading import Event
import time

from paradex.io.capture_pc.ssh import run_script
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.io.capture_pc.command_sender import CommandSender

run_script("python src/validate/command_sender/stream_client.py")
start_event = Event()
exit_event = Event()
stop_event = Event()

listen_keyboard({"c": start_event, "q": exit_event, "s": stop_event})

dc = DataCollector()
dc.start()


cs = CommandSender()

cur_state = "stop"
while not exit_event.is_set():
    if start_event.is_set() and cur_state != "start":
        cs.send_command("start")
        cur_state = "start"
        start_event.clear()
        continue
    
    if stop_event.is_set() and cur_state != "stop":
        cs.send_command("stop")
        cur_state = "stop"
        stop_event.clear()
        continue
     
    time.sleep(0.1)
    print(cur_state)
    data_dict = dc.get_data()
    for pc_id, data in data_dict.items():
        print(f"PC ID: {pc_id}, Data: {data}")

dc.end()
cs.end()