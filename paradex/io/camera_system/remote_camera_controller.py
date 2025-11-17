import zmq
from datetime import datetime
import time
from threading import Thread, Event

from paradex.utils.system import get_pc_list, get_pc_ip

class remote_camera_controller:
    def __init__(self, name, pc_list=None):
        self.name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.pc_list = get_pc_list() if pc_list is None else pc_list
        
        self.ping_port = 5480    
        self.command_port = 5482   
        self.connection_port = 5483
        
        self.exit_event = Event()
        self.start_event = Event()
        self.stop_event = Event()   
        self.sending_event = Event()
        self.error_event = Event()

        self.run_thread = Thread(target=self.run, daemon=True)
        self.run_thread.start()

        
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
            socket.setsockopt(zmq.RCVTIMEO, 1000) 
            socket.setsockopt(zmq.SNDTIMEO, 1000)
            socket.connect(f"tcp://{get_pc_ip(pc)}:{self.command_port}")
            self.command_sockets[pc] = socket
            print(f"{pc}: Command socket connected")

        if failed_pcs:
            raise ConnectionError(
                f"다음 PC들이 응답하지 않습니다: {failed_pcs}\n"
                f"각 PC에서 'python src/camera/server_daemon.py'를 실행하세요."
            )
        
        self.register()
    
    def check_server_alive(self, pc):
        """ping port로 서버 확인"""
        socket = self.ctx.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 5000)
        socket.setsockopt(zmq.SNDTIMEO, 5000)
        
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
        response = {}
        
        for pc, socket in self.command_sockets.items():
            try:
                socket.send_json(cmd)
                response[pc] = socket.recv_json()

            except zmq.ZMQError as e:
                print(f"{pc}: No response in send_command - {e}")
                response[pc] = {'status':'error', 'msg':'no response'}
                
        return response
            
    def register(self):
        cmd = {'action': 'register'}
        self.send_command(cmd)
        
    def start(self, mode, syncMode, save_path=None, fps=30):
        self.mode = mode
        self.syncMode = syncMode
        self.save_path = save_path
        self.fps = fps
        
        self.sending_event.clear()
        self.start_event.set()
        
        self.sending_event.wait()

    def stop(self):
        self.sending_event.clear()
        self.stop_event.set()
        self.sending_event.wait()

    def end(self):
        self.exit_event.set()        
        self.run_thread.join()
        
    def run(self):
        self.initialize()
        
        while not self.exit_event.is_set():
            cmd = {'action': 'heartbeat'}
            
            if self.start_event.is_set():
                cmd = {
                    'action': 'start',
                    'mode': self.mode,
                    'syncMode': self.syncMode,
                    'save_path': self.save_path,
                    'fps': self.fps
                }
                self.start_event.clear()
                
            if self.stop_event.is_set():
                cmd = {'action': 'stop'}
                self.stop_event.clear()

            response = self.send_command(cmd)
            if cmd['action'] in ['start', 'stop']:
                self.sending_event.set()
            
            for pc, resp in response.items():
                if resp['status'] == 'error':
                    print(f"{pc}: {resp['msg']}")
                    self.error_event.set()
                    
            time.sleep(0.1)
            
        self.send_command({'action': 'end'})
        for socket in self.command_sockets.values():
            socket.close()
        self.ctx.term()
    
    def is_error(self):
        return self.error_event.is_set()