import paramiko
import time
import zmq

from paradex.utils.env import get_pcinfo

class pc_state:
    DISCONNECTED = 0
    CONNECTED = 1
    RUNNING = 2

class CameraMonitor:
    def __init__(self, port=5479):
        self.port = port
        self.pc_info = get_pcinfo()
        self.pc_list = list(self.pc_info.keys())
