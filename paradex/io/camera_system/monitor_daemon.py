import time
import zmq
from flask import Flask, render_template, jsonify
from threading import Thread

from paradex.utils.env import get_pcinfo
from paradex.io.capture_pc.connect import run_script

class pc_state:
    DISCONNECTED = 0
    CONNECTED = 1
    RUNNING = 2

class CameraMonitor:
    def __init__(self, port=5479, web_port=8080, ping_interval=1.0):
        self.port = port
        self.monitor_port = 5480  # ping-pong용 포트
        self.pc_info = get_pcinfo()
        self.pc_list = list(self.pc_info.keys())
        self.web_port = web_port
        self.ping_interval = ping_interval  # ping 보내는 주기 (초)

        self.pc_state = {pc: pc_state.RUNNING for pc in self.pc_list}
        self.camera_states = {}
        self.last_ping_time = {pc: 0 for pc in self.pc_list}

        # Flask 앱 설정
        self.app = Flask(__name__)
        self.setup_routes()

        self.initialize()
        
        # 백그라운드에서 모니터링 시작
        monitor_thread = Thread(target=self.run_monitor, daemon=True)
        monitor_thread.start()
        
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
                'camera_states': self.camera_states
            })

    def initialize(self):
        """초기 서버 상태 확인 및 시작"""
        for pc in self.pc_list:
            state = self.check_server_open(pc)
            if not state:
                started = self.start_server(pc)
                if not started:
                    self.pc_state[pc] = pc_state.DISCONNECTED
                else:
                    self.pc_state[pc] = pc_state.RUNNING

    def check_server_open(self, pc):
        """서버가 열려있는지 ping으로 확인"""
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 2000)
        socket.setsockopt(zmq.SNDTIMEO, 2000)

        try:
            socket.connect(f"tcp://{self.pc_info[pc]['ip']}:{self.monitor_port}")
            socket.send_string("ping")
            response = socket.recv_string()
            socket.close()
            return response == "pong"
        
        except zmq.ZMQError:
            socket.close()
            return False
        
    def start_server(self, pc):
        """서버 데몬 시작"""
        try:
            print(f"{pc}: Starting server daemon...")
            run_script("python src/camera/server_daemon.py", [pc])
            time.sleep(2)
            
            return self.check_server_open(pc)
            
        except Exception as e:
            print(f"{pc}: Failed to start - {e}")
            return False

    def run_monitor(self):
        """백그라운드 모니터링 - 주기적 ping-pong"""
        
        while True:
            # 각 PC 상태 체크
            for pc in self.pc_list:
                alive = self.check_server_open(pc)  # 기존 함수 그대로 사용
                
                if alive:
                    if self.pc_state[pc] != pc_state.RUNNING:
                        print(f"{pc}: Server back online")
                    self.pc_state[pc] = pc_state.RUNNING
                    
                    # 카메라 상태 가져오기
                    self.get_camera_status(pc)
                else:
                    if self.pc_state[pc] == pc_state.RUNNING:
                        print(f"{pc}: Server not responding")
                    self.pc_state[pc] = pc_state.DISCONNECTED
            
            time.sleep(5.0)  # 5초마다 체크

    def get_camera_status(self, pc):
        """카메라 상태 가져오기"""
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 2000)
        
        try:
            socket.connect(f"tcp://{self.pc_info[pc]['ip']}:{self.monitor_port}")
            socket.send_json({"type": "status"})
            response = socket.recv_json()
            self.camera_states[pc] = response.get('cameras', {})
            socket.close()
        except:
            socket.close()

    def setup_ping_sockets(self):
        """각 PC별 ping 소켓 설정"""
        self.ctx = zmq.Context()
        self.ping_sockets = {}
        
        for pc in self.pc_list:
            try:
                socket = self.ctx.socket(zmq.REQ)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2초 타임아웃
                socket.setsockopt(zmq.SNDTIMEO, 2000)
                socket.connect(f"tcp://{self.pc_info[pc]['ip']}:{self.monitor_port}")
                
                self.ping_sockets[pc] = socket
                print(f"{pc}: Ping socket connected")
                
            except Exception as e:
                print(f"{pc}: Failed to setup ping socket - {e}")
                self.pc_state[pc] = pc_state.DISCONNECTED

    def ping_server(self, pc):
        """서버에 ping 보내고 상태 업데이트"""
        if pc not in self.ping_sockets:
            return
            
        socket = self.ping_sockets[pc]
        
        try:
            # status 요청 (카메라 상태까지 함께 받기)
            socket.send_json({"type": "status"})
            response = socket.recv_json()
            
            if response.get('status') == 'ok':
                # 서버 살아있음
                if self.pc_state[pc] != pc_state.RUNNING:
                    print(f"{pc}: Server back online")
                    self.pc_state[pc] = pc_state.RUNNING
                
                # 카메라 상태 업데이트
                self.camera_states[pc] = response.get('cameras', {})
                print(f"{pc}: {self.camera_states[pc]}")
            
        except zmq.Again:
            # 타임아웃 - 서버 응답 없음
            if self.pc_state[pc] == pc_state.RUNNING:
                print(f"{pc}: Server not responding (timeout)")
                self.pc_state[pc] = pc_state.DISCONNECTED
            
            # 소켓 재연결 시도
            self.reconnect_socket(pc)
            
        except Exception as e:
            print(f"{pc}: Ping failed - {e}")
            if self.pc_state[pc] == pc_state.RUNNING:
                self.pc_state[pc] = pc_state.DISCONNECTED
            
            # 소켓 재연결 시도
            self.reconnect_socket(pc)

    def reconnect_socket(self, pc):
        """소켓 재연결"""
        try:
            if pc in self.ping_sockets:
                self.ping_sockets[pc].close()
            
            socket = self.ctx.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 2000)
            socket.setsockopt(zmq.SNDTIMEO, 2000)
            socket.connect(f"tcp://{self.pc_info[pc]['ip']}:{self.monitor_port}")
            
            self.ping_sockets[pc] = socket
            
        except Exception as e:
            print(f"{pc}: Reconnect failed - {e}")
