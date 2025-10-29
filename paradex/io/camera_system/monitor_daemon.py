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
        self.setup_heartbeat()
        
        try:
            while True:
                self.receive_heartbeat()
                self.print_status()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nMonitor stopped")

    def setup_heartbeat(self):
        """Heartbeat 수신 소켓 설정"""
        self.ctx = zmq.Context()
        self.heartbeat_sockets = {}
        self.camera_states = {}  # 각 PC의 카메라 상태 저장
        
        for pc in self.pc_list:
            if self.pc_state[pc] == pc_state.RUNNING:
                try:
                    socket = self.ctx.socket(zmq.SUB)
                    socket.connect(f"tcp://{self.pc_info[pc]['ip']}:5481")  # monitor_port
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
                self.camera_states[pc] = status.get('cameras', {})
            except zmq.Again:
                pass  # No message

    def print_status(self):
        """상태 출력"""
        print("\033[2J\033[H")  # 화면 클리어
        print("=" * 80)
        print(f"Camera Monitor - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()
        
        # PC 상태
        print("PC Status:")
        print("-" * 80)
        for pc in self.pc_list:
            state = self.pc_state[pc]
            if state == pc_state.RUNNING:
                status = "✓ RUNNING"
            elif state == pc_state.CONNECTED:
                status = "○ CONNECTED"
            else:
                status = "✗ DISCONNECTED"
            
            print(f"  {pc:20s} : {status}")
        
        print()
        
        # 카메라 상태
        print("Camera Status:")
        print("-" * 80)
        print(f"  {'Camera':20s} | {'PC':20s} | {'State':12s} | {'Frame ID':10s}")
        print("-" * 80)
        
        for pc, cameras in self.camera_states.items():
            for cam_name, cam_status in cameras.items():
                state = cam_status.get('state', 'UNKNOWN')
                frame_id = cam_status.get('frame_id', 0)
                
                # 상태별 색상
                if state == "CAPTURING":
                    color = "\033[94m"  # Blue
                elif state == "READY":
                    color = "\033[92m"  # Green
                else:
                    color = "\033[93m"  # Yellow
                
                print(f"  {cam_name:20s} | {pc:20s} | {color}{state:12s}\033[0m | {frame_id:10d}")
        
        if not self.camera_states:
            print("  No camera data received yet...")
        
        print()
        print("=" * 80)
        print("Press Ctrl+C to exit")
        print("=" * 80)