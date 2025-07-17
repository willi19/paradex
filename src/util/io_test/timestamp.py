import time
from threading import Event

from paradex.io.camera.timecode_receiver import TimecodeReceiver
from paradex.utils.keyboard_listener import listen_keyboard

stop_event = Event()
listen_keyboard({"q":stop_event})

tr = TimecodeReceiver()
tr.start()

while not stop_event.is_sewt():
    print(tr.get_data())
    time.sleep(0.1)

tr.end()
tr.quit()
