"""
Camera Monitor Daemon (Main PC)

카메라 시스템 모니터링 전용
- 각 PC 연결 상태
- Camera worker (camera_server) 실행 상태
- 각 카메라 상태 및 Frame ID
- Sync 문제 감지
- 실시간 시각화

역할: 모니터링만 (제어는 RemoteCameraController 사용)

위치: paradex/io/camera_system/monitor_daemon.py
실행: python -m paradex.io.camera_system.monitor_daemon
"""

import time
import threading
import logging
import signal
import sys
import subprocess
import zmq
import json
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

from paradex.utils.env import get_pcinfo
from paradex.utils.file_io import log_path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_path}/camera_monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CameraState:
    """카메라 상태 정의"""
    UNKNOWN = "UNKNOWN"
    DISCONNECTED = "DISCONNECTED"
    READY = "READY"
    CAPTURING = "CAPTURING"
    ERROR = "ERROR"


class PCStatus:
    """PC 상태"""
    def __init__(self, pc_name: str):
        self.pc_name = pc_name
        self.connected = False
        self.camera_server_running = False
        self.last_check_time = None


class CameraStatus:
    """카메라 상태"""
    def __init__(self, serial_num: str):
        self.serial_num = serial_num
        self.state = CameraState.UNKNOWN
        self.frame_id = 0
        self.last_update_time = None
        self.pc_name = None


class CameraMonitorDaemon:    
    def __init__(self, update_interval: float = 1.0):
        """
        Args:
            update_interval: 상태 업데이트 주기 (초)
        """
        self.update_interval = update_interval
        
        # PC 정보
        self.pc_info = get_pcinfo()
        self.pc_list = list(self.pc_info.keys())
        
        # 상태 저장
        self.pc_status: Dict[str, PCStatus] = {}
        self.camera_status: Dict[str, CameraStatus] = {}
        
        # PC별 카메라 매핑
        self.pc_cameras: Dict[str, List[str]] = {}
        
        # 초기화
        self._initialize_status()
        
        # 모니터링 스레드
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        
        # Heartbeat 수신용 ZMQ
        self.heartbeat_context = zmq.Context()
        self.heartbeat_sockets: Dict[str, zmq.Socket] = {}
        self._setup_heartbeat_sockets()
        
        # Signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Camera Monitor Daemon created")
        logger.info(f"Monitoring {len(self.pc_list)} PCs: {self.pc_list}")
    
    def _initialize_status(self):
        """상태 초기화"""
        # PC 상태
        for pc_name in self.pc_list:
            self.pc_status[pc_name] = PCStatus(pc_name)
        
        # 카메라 상태
        for pc_name, info in self.pc_info.items():
            cam_list = info.get('cam_list', [])
            self.pc_cameras[pc_name] = cam_list
            
            for serial_num in cam_list:
                camera = CameraStatus(serial_num)
                camera.pc_name = pc_name
                self.camera_status[serial_num] = camera
    
    def _setup_heartbeat_sockets(self):
        """Heartbeat 수신용 소켓 설정"""
        heartbeat_port = 5565
        
        for pc_name in self.pc_list:
            try:
                pc_ip = self.pc_info[pc_name]['ip']
                
                socket = self.heartbeat_context.socket(zmq.SUB)
                socket.connect(f"tcp://{pc_ip}:{heartbeat_port}")
                socket.setsockopt_string(zmq.SUBSCRIBE, '')  # 모든 메시지 수신
                socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
                
                self.heartbeat_sockets[pc_name] = socket
                logger.info(f"Heartbeat socket connected to {pc_name} ({pc_ip}:{heartbeat_port})")
                
            except Exception as e:
                logger.error(f"Failed to setup heartbeat socket for {pc_name}: {e}")
    
    def _signal_handler(self, signum, frame):
        """Signal handler"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """모니터링 시작"""
        if self.running:
            logger.warning("Already running")
            return
        
        logger.info("=" * 80)
        logger.info("Starting Camera Monitor Daemon")
        logger.info("=" * 80)
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=False)
        self.monitor_thread.start()
        
        # Heartbeat 수신 스레드
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=False)
        self.heartbeat_thread.start()
        
        logger.info("Monitor started")
    
    def stop(self):
        """모니터링 중지"""
        if not self.running:
            return
        
        logger.info("Stopping monitor...")
        self.running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5)
        
        # Heartbeat 소켓 정리
        for socket in self.heartbeat_sockets.values():
            socket.close()
        self.heartbeat_context.term()
        
        logger.info("Monitor stopped")
    
    def _monitor_loop(self):
        """모니터링 메인 루프"""
        logger.info("Monitor loop started")
        
        while self.running:
            try:
                # 1. PC 연결 상태 체크
                self._check_pc_connections()
                
                # 2. Camera server 실행 상태 체크
                self._check_camera_servers()
                
                # 3. 시각화
                self._print_status()
                
                # 4. 문제 감지
                self._detect_issues()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}", exc_info=True)
                time.sleep(self.update_interval)
        
        logger.info("Monitor loop stopped")
    
    def _heartbeat_loop(self):
        """Heartbeat 수신 루프"""
        logger.info("Heartbeat loop started")
        
        while self.running:
            try:
                # 각 PC의 heartbeat 수신
                for pc_name, socket in self.heartbeat_sockets.items():
                    try:
                        # Non-blocking receive
                        message = socket.recv_string(flags=zmq.NOBLOCK)
                        
                        # Parse JSON
                        status = json.loads(message)
                        
                        # 카메라 상태 업데이트
                        self._update_camera_status(pc_name, status)
                        
                    except zmq.Again:
                        # No message available
                        pass
                    except Exception as e:
                        logger.debug(f"Heartbeat receive error from {pc_name}: {e}")
                
                time.sleep(0.1)  # 100ms
                
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}", exc_info=True)
                time.sleep(0.1)
        
        logger.info("Heartbeat loop stopped")
    
    def _update_camera_status(self, pc_name: str, status: dict):
        """Heartbeat 데이터로 카메라 상태 업데이트"""
        cameras_data = status.get('cameras', {})
        
        for serial_num, cam_data in cameras_data.items():
            if serial_num in self.camera_status:
                camera = self.camera_status[serial_num]
                
                # 상태 업데이트
                camera.state = cam_data.get('state', CameraState.UNKNOWN)
                camera.frame_id = cam_data.get('frame_id', 0)
                camera.last_update_time = time.time()
                
                # 에러 정보
                if 'error' in cam_data:
                    logger.warning(f"Camera {serial_num} error: {cam_data['error']}")
    
    def _check_pc_connections(self):
        """PC 연결 상태 체크 (ping)"""
        for pc_name, status in self.pc_status.items():
            try:
                pc_ip = self.pc_info[pc_name]['ip']
                
                # ping으로 연결 체크
                result = subprocess.run(
                    ['ping', '-c', '1', '-W', '1', pc_ip],
                    capture_output=True,
                    timeout=2
                )
                
                status.connected = (result.returncode == 0)
                status.last_check_time = time.time()
                
            except Exception as e:
                status.connected = False
                logger.debug(f"PC {pc_name} connection check failed: {e}")
    
    def _check_camera_servers(self):
        """Camera server 실행 상태 체크 (SSH)"""
        for pc_name, status in self.pc_status.items():
            if not status.connected:
                status.camera_server_running = False
                continue
            
            try:
                pc_ip = self.pc_info[pc_name]['ip']
                user = self.pc_info[pc_name].get('user', 'robot')
                
                # SSH로 프로세스 체크
                result = subprocess.run(
                    ['ssh', f'{user}@{pc_ip}', 'pgrep -f camera_server.py'],
                    capture_output=True,
                    timeout=2
                )
                
                pids = result.stdout.decode().strip()
                status.camera_server_running = bool(pids)
                
            except Exception as e:
                status.camera_server_running = False
                logger.debug(f"Camera server check failed for {pc_name}: {e}")
    
    def _print_status(self):
        """상태 출력"""
        # 화면 클리어 (ANSI escape code)
        print("\033[2J\033[H", end='')
        
        print("=" * 80)
        print(f"Camera Monitor Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()
        
        # PC 상태
        print("PC Status:")
        print("-" * 80)
        for pc_name, status in self.pc_status.items():
            conn_icon = "✓" if status.connected else "✗"
            server_icon = "✓" if status.camera_server_running else "✗"
            
            print(f"  {pc_name:15s} | Connection: {conn_icon} | Camera Server: {server_icon}")
        
        print()
        
        # 카메라 상태
        print("Camera Status:")
        print("-" * 80)
        print(f"  {'Serial':20s} | {'PC':15s} | {'State':12s} | {'Frame ID':10s}")
        print("-" * 80)
        
        # PC별로 그룹화
        for pc_name in self.pc_list:
            cameras = [c for c in self.camera_status.values() if c.pc_name == pc_name]
            
            if cameras:
                for camera in cameras:
                    state_color = self._get_state_color(camera.state)
                    print(f"  {camera.serial_num:20s} | {pc_name:15s} | {state_color}{camera.state:12s}\033[0m | {camera.frame_id:10d}")
        
        print()
        
        # Sync 상태
        print("Sync Status:")
        print("-" * 80)
        frame_ids = [c.frame_id for c in self.camera_status.values() if c.state == CameraState.CAPTURING]
        
        if len(frame_ids) > 1:
            max_diff = max(frame_ids) - min(frame_ids)
            if max_diff > 10:
                print(f"  ⚠️  SYNC WARNING: Frame ID difference = {max_diff}")
            else:
                print(f"  ✓ Sync OK (diff = {max_diff})")
        else:
            print("  - Not capturing or insufficient cameras")
        
        print()
        print("=" * 80)
        print("Press Ctrl+C to exit")
        print("=" * 80)
    
    def _get_state_color(self, state: str) -> str:
        """상태별 색상 코드"""
        colors = {
            CameraState.READY: "\033[92m",        # Green
            CameraState.CAPTURING: "\033[94m",    # Blue
            CameraState.ERROR: "\033[91m",        # Red
            CameraState.DISCONNECTED: "\033[90m", # Gray
            CameraState.UNKNOWN: "\033[93m",      # Yellow
        }
        return colors.get(state, "")
    
    def _detect_issues(self):
        """문제 감지 및 로깅"""
        # PC 연결 끊김
        for pc_name, status in self.pc_status.items():
            if not status.connected:
                logger.warning(f"PC {pc_name} is disconnected")
        
        # Camera server 죽음
        for pc_name, status in self.pc_status.items():
            if status.connected and not status.camera_server_running:
                logger.warning(f"Camera server not running on {pc_name}")
        
        # Sync 문제
        capturing_cameras = [c for c in self.camera_status.values() if c.state == CameraState.CAPTURING]
        if len(capturing_cameras) > 1:
            frame_ids = [c.frame_id for c in capturing_cameras]
            max_diff = max(frame_ids) - min(frame_ids)
            
            if max_diff > 10:
                logger.warning(f"Sync issue detected: Frame ID diff = {max_diff}")
                for camera in capturing_cameras:
                    logger.warning(f"  {camera.serial_num}: Frame {camera.frame_id}")
    
    def run(self):
        """실행 (blocking)"""
        try:
            self.start()
            
            # 메인 스레드는 대기
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
