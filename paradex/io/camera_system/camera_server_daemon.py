import threading
import zmq
import time

from paradex.io.camera_system.camera_loader import CameraLoader

class camera_server_daemon:
    def __init__(self):
        self.camera_loader = CameraLoader()

        self.ping_port = 5480
        self.command_port = 5481
        self.ctx = zmq.Context()

        threading.Thread(target=self.pingpong_thread, daemon=True).start()
        threading.Thread(target=self.monitor_thread, daemon=True).start()
        threading.Thread(target=self.command_thread, daemon=True).start()

    def pingpong_thread(self):
        self.ping_socket = self.ctx.socket(zmq.REP)
        self.ping_socket.setsockopt(zmq.LINGER, 0)
        self.ping_socket.bind(f"tcp://*:{self.ping_port}")

        while True:
            try:
                _ = self.ping_socket.recv_string(flags=zmq.NOBLOCK)
                self.ping_socket.send_string("pong")
            except zmq.ZMQError:
                time.sleep(0.1)

    def monitor_thread(self):
        while True:
            pass

    def command_thread(self):
        self.command_socket = zmq.Context().socket(zmq.REP)
        while True:
            try:
                self.command_socket.bind(f"tcp://*:{self.command_port}")
                break
            except zmq.ZMQError:
                pass