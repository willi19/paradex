import time
import threading

from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.utils.system import network_info
from paradex.utils.keyboard_listener import listen_keyboard

end_event = threading.Event()
listen_keyboard({"q": end_event})
ts = TimestampMonitor(**network_info.get("timestamp")["param"])
ts.start("tmp")
# start_time = time.time()
# last_frame_id = 0
# while time.time() - start_time < 10.0:
#     data = ts.get_data()
#     if data['frame_id'] != last_frame_id:
#         last_frame_id = data['frame_id']
#         print(ts.get_data()['frame_id'], ": ", ts.get_data()['time'] - start_time)
# ts.stop()

# ts.wait_signal_inactive(30)
time.sleep(1.0)
ts.stop()
ts.end()