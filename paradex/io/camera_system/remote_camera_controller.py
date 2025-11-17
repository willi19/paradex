import zmq
from datetime import datetime

from paradex.utils.system import get_pc_list, get_pc_ip

class remote_camera_controller:
    def __init__(self, name, pc_list=None):
        self.name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.pc_list = get_pc_list() if pc_list is None else pc_list
        
        self.ping_port = 5480    
        self.command_port = 5482   
        
        self.initialize()

    def initialize(self):
        self.ctx = zmq.Context()
        self.command_sockets = {}
        failed_pcs = []

        for pc in self.pc_list:
            if not self.check_server_alive(pc):
                failed_pcs.append(pc)
                continue
            
            socket = self.ctx.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 60000) 
            socket.setsockopt(zmq.SNDTIMEO, 60000)
            socket.connect(f"tcp://{get_pc_ip(pc)}:{self.command_port}")
            self.command_sockets[pc] = socket
            print(f"{pc}: Command socket connected")

        if failed_pcs:
            raise ConnectionError(
                f"다음 PC들이 응답하지 않습니다: {failed_pcs}\n"
                f"각 PC에서 'python src/camera/server_daemon.py'를 실행하세요."
            )
    
    def check_server_alive(self, pc):
        """ping port로 서버 확인"""
        socket = self.ctx.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 2000)
        socket.setsockopt(zmq.SNDTIMEO, 2000)
        
        try:
            socket.connect(f"tcp://{get_pc_ip(pc)}:{self.ping_port}")
            socket.send_string("ping")
            response = socket.recv_string()
            return response == "pong"
        except zmq.ZMQError:
            return False
        finally:
            socket.close()
    
    def send_command(self, cmd):
        """명령 전송 및 응답 수신"""
        cmd['controller_name'] = self.name
        for pc, socket in self.command_sockets.items():
            try:
                socket.send_json(cmd)
                response = socket.recv_json()
                print(f"{pc}: {response}")
            except zmq.ZMQError as e:
                print(f"{pc}: No response - {e}")
            
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