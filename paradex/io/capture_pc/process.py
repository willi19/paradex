# paradex/video/distributed_processor.py
import cv2
import os
import json
import time
import socket
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Optional
import requests
from threading import Thread
from tqdm import tqdm

@dataclass
class ProcessStatus:
    """단일 비디오 처리 상태"""
    video_path: str
    pc_name: str
    total_frames: int
    current_frame: int
    fps: float
    status: str  # 'processing', 'done', 'error'
    error_msg: Optional[str] = None
    start_time: Optional[float] = None
    
    @property
    def progress(self):
        return self.current_frame / self.total_frames if self.total_frames > 0 else 0
    
    @property
    def eta_seconds(self):
        if self.fps > 0:
            return (self.total_frames - self.current_frame) / self.fps
        return None
    
    def to_dict(self):
        return asdict(self)

class StatusServer:
    """
    중앙 서버 - 각 PC의 처리 상태 수집
    
    Example:
        >>> server = StatusServer(port=5000)
        >>> server.start()
        >>> # 웹 브라우저에서 http://central-pc:5000 접속
    """
    
    def __init__(self, port=5000):
        self.port = port
        self.status_db: Dict[str, ProcessStatus] = {}
        self.running = False
    
    def start(self):
        """Flask 서버 시작"""
        from flask import Flask, jsonify, render_template_string
        
        app = Flask(__name__)
        self.app = app
        
        @app.route('/status')
        def get_status():
            """JSON API"""
            return jsonify({
                pc_name: status.to_dict() 
                for pc_name, status in self.status_db.items()
            })
        
        @app.route('/')
        def dashboard():
            """웹 대시보드"""
            return render_template_string(DASHBOARD_HTML)
        
        @app.route('/update', methods=['POST'])
        def update_status():
            """Worker가 상태 업데이트"""
            from flask import request
            data = request.json
            
            status = ProcessStatus(**data)
            self.status_db[status.pc_name] = status
            
            return jsonify({'success': True})
        
        print(f"🖥️  Status server starting at http://0.0.0.0:{self.port}")
        print(f"   Access from browser: http://{socket.gethostname()}:{self.port}")
        
        self.running = True
        app.run(host='0.0.0.0', port=self.port, debug=False)
    
    def stop(self):
        self.running = False

class WorkerProcessor:
    """
    Worker PC - 비디오 처리하면서 중앙 서버에 상태 보고
    
    Example:
        >>> worker = WorkerProcessor(
        ...     central_server="http://192.168.1.100:5000",
        ...     pc_name="capture-pc-1"
        ... )
        >>> worker.process_video("input.mp4", "output.mp4", my_process_fn)
    """
    
    def __init__(self, central_server: str, pc_name: Optional[str] = None):
        self.central_server = central_server
        self.pc_name = pc_name or socket.gethostname()
        self.status = None
    
    def _report_status(self):
        """중앙 서버에 상태 보고"""
        if self.status is None:
            return
        
        try:
            requests.post(
                f"{self.central_server}/update",
                json=self.status.to_dict(),
                timeout=1
            )
        except Exception as e:
            print(f"⚠️  Failed to report status: {e}")
    
    def process_video(self, video_path, output_path, process_fn, report_interval=1.0):
        """
        비디오 처리하면서 실시간 상태 보고
        
        Args:
            video_path: 입력 비디오
            output_path: 출력 비디오
            process_fn: 처리 함수(frame, fid) -> processed_frame
            report_interval: 상태 보고 간격 (초)
        """
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 출력 준비
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # 초기 상태
        self.status = ProcessStatus(
            video_path=video_path,
            pc_name=self.pc_name,
            total_frames=total_frames,
            current_frame=0,
            fps=0,
            status='processing',
            start_time=time.time()
        )
        
        last_report = time.time()
        frame_times = []
        
        try:
            for fid in tqdm(range(total_frames), desc=f"[{self.pc_name}] Processing"):
                frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 프레임 처리
                processed = process_fn(frame, fid)
                out.write(processed)
                
                # FPS 계산
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                
                # 상태 업데이트
                self.status.current_frame = fid + 1
                self.status.fps = 1.0 / (sum(frame_times) / len(frame_times))
                
                # 주기적으로 중앙 서버에 보고
                if time.time() - last_report > report_interval:
                    self._report_status()
                    last_report = time.time()
            
            # 완료
            self.status.status = 'done'
            self.status.current_frame = total_frames
            self._report_status()
            
            print(f"✅ [{self.pc_name}] Completed: {output_path}")
            
        except Exception as e:
            self.status.status = 'error'
            self.status.error_msg = str(e)
            self._report_status()
            print(f"❌ [{self.pc_name}] Error: {e}")
            raise
        
        finally:
            cap.release()
            out.release()
    
    def process_batch(self, video_paths, output_dir, process_fn):
        """여러 비디오 순차 처리"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for video_path in video_paths:
            video_name = Path(video_path).name
            output_path = Path(output_dir) / video_name
            self.process_video(video_path, str(output_path), process_fn)

# 웹 대시보드 HTML
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Video Processing Monitor</title>
    <style>
        body { 
            font-family: Arial; 
            margin: 20px;
            background: #1e1e1e;
            color: #fff;
        }
        .status-card {
            background: #2d2d2d;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #4CAF50;
        }
        .status-card.error { border-left-color: #f44336; }
        .status-card.done { border-left-color: #2196F3; }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #444;
            border-radius: 5px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin: 10px 0;
        }
        .stat-item {
            background: #3d3d3d;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        .stat-label { 
            font-size: 12px; 
            color: #888; 
            margin-bottom: 5px;
        }
        .stat-value { 
            font-size: 20px; 
            font-weight: bold; 
        }
        h1 { text-align: center; }
        .refresh-info {
            text-align: center;
            color: #888;
            margin: 20px 0;
        }
    </style>
    <script>
        async function updateStatus() {
            const response = await fetch('/status');
            const data = await response.json();
            
            const container = document.getElementById('status-container');
            container.innerHTML = '';
            
            for (const [pc_name, status] of Object.entries(data)) {
                const progress = status.current_frame / status.total_frames * 100;
                const eta = status.eta_seconds ? Math.floor(status.eta_seconds) : 0;
                
                const card = document.createElement('div');
                card.className = `status-card ${status.status}`;
                card.innerHTML = `
                    <h3>🖥️ ${pc_name}</h3>
                    <p><strong>Video:</strong> ${status.video_path}</p>
                    
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progress}%">
                            ${progress.toFixed(1)}%
                        </div>
                    </div>
                    
                    <div class="stats">
                        <div class="stat-item">
                            <div class="stat-label">Frames</div>
                            <div class="stat-value">${status.current_frame} / ${status.total_frames}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Speed</div>
                            <div class="stat-value">${status.fps.toFixed(1)} fps</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">ETA</div>
                            <div class="stat-value">${eta}s</div>
                        </div>
                    </div>
                    
                    <p><strong>Status:</strong> ${status.status.toUpperCase()}</p>
                    ${status.error_msg ? `<p style="color: #f44336"><strong>Error:</strong> ${status.error_msg}</p>` : ''}
                `;
                
                container.appendChild(card);
            }
        }
        
        // 1초마다 자동 갱신
        setInterval(updateStatus, 1000);
        updateStatus();
    </script>
</head>
<body>
    <h1>📊 Video Processing Monitor</h1>
    <div class="refresh-info">Auto-refresh every 1 second</div>
    <div id="status-container"></div>
</body>
</html>
"""

# 간단한 CLI 모니터 (웹 대신)
class CLIMonitor:
    """터미널에서 상태 모니터링"""
    
    def __init__(self, central_server: str):
        self.central_server = central_server
        self.running = False
    
    def start(self):
        """실시간 상태 출력"""
        self.running = True
        
        print("📊 Video Processing Monitor")
        print("=" * 80)
        
        try:
            while self.running:
                os.system('clear' if os.name != 'nt' else 'cls')
                
                try:
                    response = requests.get(f"{self.central_server}/status", timeout=2)
                    statuses = response.json()
                    
                    print("\n📊 Video Processing Monitor")
                    print("=" * 80)
                    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print()
                    
                    for pc_name, status_dict in statuses.items():
                        status = ProcessStatus(**status_dict)
                        progress = status.progress * 100
                        
                        bar_width = 40
                        filled = int(bar_width * status.progress)
                        bar = '█' * filled + '░' * (bar_width - filled)
                        
                        print(f"🖥️  {pc_name}")
                        print(f"   Video: {Path(status.video_path).name}")
                        print(f"   [{bar}] {progress:.1f}%")
                        print(f"   {status.current_frame}/{status.total_frames} frames")
                        print(f"   Speed: {status.fps:.1f} fps, ETA: {status.eta_seconds:.0f}s")
                        print(f"   Status: {status.status.upper()}")
                        if status.error_msg:
                            print(f"   ❌ Error: {status.error_msg}")
                        print()
                    
                except requests.RequestException as e:
                    print(f"⚠️  Cannot connect to server: {e}")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitor stopped")
            self.running = False