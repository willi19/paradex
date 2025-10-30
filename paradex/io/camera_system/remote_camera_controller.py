import zmq
from datetime import datetime

from paradex.utils.env import get_pcinfo

class remote_camera_controller:
    def __init__(self, name, pc_list=None):
        self.name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.pc_info = get_pcinfo()
        self.pc_list = pc_list if pc_list is not None else list(self.pc_info.keys())
        self.command_port = 5482
        
        self.initialize()

    def initialize(self):
        self.ctx = zmq.Context()
        self.command_sockets = {}

        for pc in self.pc_list:
            socket = self.ctx.socket(zmq.REQ)
            socket.setsockopt(zmq.RCVTIMEO, self.command_port)
            socket.connect(f"tcp://{self.pc_info[pc]['ip']}:{self.command_port}")
            self.command_sockets[pc] = socket
    
    def send_command(self, cmd):
        """명령 전송 및 응답 수신"""
        cmd['controller_name'] = self.name
        for pc, socket in self.command_sockets.items():
            socket.send_json(cmd)
            try:
                response = socket.recv_json()
                print(f"{pc}: {response}")
            except zmq.ZMQError:
                print(f"{pc}: No response received.")
            
    def start(self, mode, syncMode, save_path=None, fps=30):
        cmd = {
            'action': 'start',
            'mode': mode,
            'syncMode': syncMode,
            'save_path': save_path,
            'fps': fps
        }
        self.send_command(cmd)

    def stop(self):
        cmd = {'action': 'stop'}
        self.send_command(cmd)

    def end(self):
        cmd = {'action': 'exit'}
        self.send_command(cmd)
        
        for socket in self.command_sockets.values():
            socket.close()
        self.ctx.term()