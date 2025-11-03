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
    def __init__(self, port=5479, web_port=8080):
        self.port = port
        self.monitor_port = 5481    
        self.pc_info = get_pcinfo()
        self.pc_list = list(self.pc_info.keys())
        self.web_port = web_port

        self.pc_state = {pc: pc_state.RUNNING for pc in self.pc_list}
        self.camera_states = {}

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
            time.sleep(2)
            
            return self.check_server_open(pc)
            
        except Exception as e:
            print(f"{pc}: Failed to start - {e}")
            return False

    def run_monitor(self):
        """백그라운드 모니터링"""
        self.setup_heartbeat()
        
        while True:
            self.receive_heartbeat()
            time.sleep(0.01)

    def setup_heartbeat(self):
        """Heartbeat 수신 소켓 설정"""
        self.ctx = zmq.Context()
        self.heartbeat_sockets = {}
        
        for pc in self.pc_list:
            if self.pc_state[pc] == pc_state.RUNNING:
                try:
                    socket = self.ctx.socket(zmq.SUB)
                    socket.connect(f"tcp://{self.pc_info[pc]['ip']}:{self.monitor_port}")
                    socket.setsockopt_string(zmq.SUBSCRIBE, '')
                    socket.setsockopt(zmq.RCVTIMEO, 100)
                    
                    self.heartbeat_sockets[pc] = socket
                except Exception as e:
                    print(f"{pc}: Failed to setup heartbeat - {e}")

    def receive_heartbeat(self):
        """Heartbeat 수신"""
        for pc, socket in self.heartbeat_sockets.items():
            try:
                status = socket.recv_json(flags=zmq.NOBLOCK)
                print(pc)
                self.camera_states[pc] = status.get('cameras', {})
            except zmq.Again:
                pass