import paramiko
import time
import zmq

from paradex.utils.env import get_pcinfo
from paradex.io.capture_pc.connect import run_script

class pc_state:
    DISCONNECTED = 0
    CONNECTED = 1
    RUNNING = 2

class CameraMonitor:
    def __init__(self, port=5479):
        self.port = port
        self.pc_info = get_pcinfo()
        self.pc_list = list(self.pc_info.keys())

        self.pc_state = {pc: pc_state.CONNECTED for pc in self.pc_list}

        self.initialize()
        self.run()

    def initialize(self):
        print(self.pc_list)
        for pc in self.pc_list:
            state = self.check_server_open(pc)
            if not state:
                started = self.start_server(pc)
                if not started:
                    self.pc_state[pc] = pc_state.DISCONNECTED

    def check_server_open(self, pc):
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 2000)
        socket.setsockopt(zmq.SNDTIMEO, 2000)

        try:
            socket.connect(f"tcp://{self.pc_info[pc]['ip']}:{5480}")
            socket.send_string("ping")
            _ = socket.recv_string()
            socket.close()
            return True
        
        except zmq.ZMQError:
            socket.close()
            return False
        
    def start_server(self, pc):
        try:
            print(f"{pc}: Starting server daemon...")
            run_script("python src/camera/server_daemon.py", [pc])
            time.sleep(2)  # 시작 대기
            
            return self.check_server_open(pc)
            
        except Exception as e:
            print(f"{pc}: Failed to start - {e}")
            return False

    def run(self):
        for pc in self.pc_list:
            print(f"{pc}: {self.pc_state[pc]}")
        while True:
            time.sleep(5)