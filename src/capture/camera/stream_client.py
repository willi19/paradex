from threading import Event
import time

from paradex.io.camera_system.camera_reader import CameraReader
from paradex.io.capture_pc.data_sender import DataPublisher
from paradex.io.capture_pc.command_sender import CommandSender

dp = DataPublisher()
cs = CommandSender() 

start_event = Event()
exit_event = Event()