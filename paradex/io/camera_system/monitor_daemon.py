import time
import zmq
from flask import Flask, render_template, jsonify
from threading import Thread

from paradex.utils.system import get_pc_list, get_pc_ip
from paradex.io.capture_pc.ssh import run_script

class pc_state:
    DISCONNECTED = 0
    CONNECTED = 1
    RUNNING = 2

class CameraMonitor:
    def __init__(self, web_port=8080, ping_interval=1.0):
        self.ping_port = 5480       # ping-pong
        self.monitor_port = 5481    # SUB로 상태 받기
        self.command_port = 5482    # command (사용 안 함)
        
        self.pc_list = get_pc_list()
        self.web_port = web_port
        self.ping_interval = ping_interval
        
        self.pc_state = {pc: pc_state.DISCONNECTED for pc in self.pc_list}
        self.camera_states = {pc: {} for pc in self.pc_list}
        self.controller_states = {pc: 'None' for pc in self.pc_list}  # 추가
        
        self.ctx = zmq.Context()
        self.monitor_sockets = {}
        
        # Flask 앱 설정
        self.app = Flask(__name__)
        self.setup_routes()
        
        # 초기화
        self.initialize()
        self.setup_monitor_sockets()
        
        # 백그라운드 모니터링 시작
        Thread(target=self.ping_loop, daemon=True).start()
        Thread(target=self.monitor_loop, daemon=True).start()
        
        # 웹 서버 시작
        print(f"Starting web dashboard at http://localhost:{self.web_port}")
        self.app.run(host='0.0.0.0', port=self.web_port, debug=False)
    
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('monitor.html')
        
        @self.app.route('/api/status')
        def get_status():
            return jsonify({
                'timestamp': time.time(),
                'pc_states': {pc: state for pc, state in self.pc_state.items()},
                'camera_states': self.camera_states,
                'controller_states': self.controller_states  # 추가
            })
    
    def initialize(self):
        """초기 서버 확인 및 시작"""
        for pc in self.pc_list:
            if self.ping_server(pc):
                self.pc_state[pc] = pc_state.RUNNING
                print(f"{pc}: Server already running")
            else:
                print(f"{pc}: Starting server...")
                if self.start_server(pc):
                    self.pc_state[pc] = pc_state.RUNNING
                else:
                    self.pc_state[pc] = pc_state.DISCONNECTED
    
    def ping_server(self, pc):
        """서버 ping 확인 (5480)"""
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
    
    def start_server(self, pc):
        """서버 데몬 시작"""
        try:
            run_script("python src/camera/server_daemon.py", [pc])
            time.sleep(2)
            return self.ping_server(pc)
        except Exception as e:
            print(f"{pc}: Failed to start - {e}")
            return False
    
    def setup_monitor_sockets(self):
        """5481 SUB 소켓 구독"""
        for pc in self.pc_list:
            try:
                socket = self.ctx.socket(zmq.SUB)
                socket.connect(f"tcp://{get_pc_ip(pc)}:{self.monitor_port}")
                socket.setsockopt_string(zmq.SUBSCRIBE, '')
                socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
                self.monitor_sockets[pc] = socket
                print(f"{pc}: Monitor socket connected")
            except Exception as e:
                print(f"{pc}: Failed to setup monitor - {e}")
    
    def ping_loop(self):
        """주기적으로 ping 확인"""
        while True:
            for pc in self.pc_list:
                alive = self.ping_server(pc)
                
                if alive:
                    if self.pc_state[pc] != pc_state.RUNNING:
                        print(f"{pc}: Server back online")
                    self.pc_state[pc] = pc_state.RUNNING
                else:
                    if self.pc_state[pc] == pc_state.RUNNING:
                        print(f"{pc}: Server not responding")
                    self.pc_state[pc] = pc_state.DISCONNECTED
            
            time.sleep(self.ping_interval)
    
    def monitor_loop(self):
        """5481에서 카메라 상태 받기"""
        while True:
            for pc, socket in self.monitor_sockets.items():
                try:
                    status = socket.recv_json(flags=zmq.NOBLOCK)
                    self.camera_states[pc] = status.get('cameras', [])  # [] 로 수정
                    self.controller_states[pc] = status.get('controller', 'None')
                except zmq.Again:
                    pass
                except Exception as e:
                    print(f"{pc}: Monitor error - {e}", flush=True)
            
            time.sleep(0.1)
