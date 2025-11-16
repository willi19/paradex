# requirements: pip install pyzmq

import zmq
import json
import time
import threading
from datetime import datetime
import cv2
import numpy as np
import os
from flask import Flask, render_template_string
import webbrowser

from multiprocessing import Pool, shared_memory, Manager, Value
from paradex.utils.file_io import home_path
from paradex.video.raw_video import *
from paradex.video.raw_video_processor import RawVideoProcessor
from paradex.utils.env import get_pcinfo, get_network_info
from paradex.io.capture_pc.util import get_client_socket, get_server_socket

class RawVideoProcessorWithProgress():
    """Extended RawVideoProcessor with ZMQ progress reporting"""
    
    def __init__(self, process_frame, load_info, process_result, update_interval=1.0):
        self.update_interval = update_interval
        self.start_time = time.time()
        
        # Start progress monitoring thread
        self.process_frame = process_frame
        self.load_info = load_info
        self.process_result = process_result

    
    def start(self):
        self._monitor_progress()
        
    def register(self):
        print("start register")
        ident, msg = self.socket.recv_multipart()
        msg = msg.decode()
        if msg == "register":
            self.ident = ident
        self.send_message("registered")  
        
    def send_message(self, message):
        if isinstance(message, dict):
            message = json.dumps(message)
    
        self.socket.send_multipart([self.ident, message.encode('utf-8')])
        
    def _monitor_progress(self):
        """Monitor and send progress updates"""
        self.processor = RawVideoProcessor(save_path="", process_result=self.process_result, process_frame=self.process_frame, load_info=self.load_info, preserve=False)
        port = get_network_info()["remote_camera"]
        self.socket = get_server_socket(port)
        
        self.register()
        progress_data = self._get_progress_data()
        self.send_message(progress_data)
        
        while not self.processor.finished():
            progress_data = self._get_progress_data()
            self.send_message(progress_data)
            time.sleep(self.update_interval)
        
        self.send_message({"event":"end"})
        
    def _get_progress_data(self):
        """Get current progress data"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate total progress
        total_frames = sum(self.processor.total_frame.values())
        processed_frames = sum(counter.value for counter in self.processor.frame_counter.values())
        
        progress_percent = (processed_frames / total_frames * 100) if total_frames > 0 else 0
        
        # Calculate per-video progress
        video_progress = {}
        for vid_path in self.processor.valid_video_path_list:
            vid_total = self.processor.total_frame[vid_path]
            vid_processed = self.processor.frame_counter[vid_path].value
            vid_percent = (vid_processed / vid_total * 100) if vid_total > 0 else 0
            
            video_progress[vid_path] = {
                'processed_frames': vid_processed,
                'total_frames': vid_total,
                'progress_percent': round(vid_percent, 2)
            }
        
        # Estimate remaining time
        fps = processed_frames / elapsed_time if elapsed_time > 0 else 0
        remaining_frames = total_frames - processed_frames
        eta_seconds = remaining_frames / fps if fps > 0 else 0
        print(self.processor.log)
        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'processing' if not self.processor.finished() else 'completed',
            'overall_progress': {
                'processed_frames': processed_frames,
                'total_frames': total_frames,
                'progress_percent': round(progress_percent, 2),
                'elapsed_time': round(elapsed_time, 2),
                'eta_seconds': round(eta_seconds, 2),
                'fps': round(fps, 2)
            },
            'video_progress': video_progress,
            'log_messages': self.processor.log[-10:],  # Last 10 log messages
            'active_videos': len(self.processor.valid_video_path_list),
            'save_path': self.processor.save_path
        }
    
    def async_callback(self, result):
        """Override to include progress update"""
        super().async_callback(result)
        # Send immediate update when a video completes
        progress_data = self._get_progress_data()
        progress_data['event'] = 'video_completed'
        self.publisher.send_progress(progress_data)
    
    def error_callback(self, e):
        """Override to include error in progress"""
        super().error_callback(e)
        # Send immediate error update
        progress_data = self._get_progress_data()
        progress_data['event'] = 'error'
        progress_data['error'] = str(e)
        self.publisher.send_progress(progress_data)
    
    def close(self):
        """Clean shutdown"""
        self.monitoring = False
        if self.progress_thread.is_alive():
            self.progress_thread.join(timeout=2)
        
        # Send final completion message
        final_data = self._get_progress_data()
        final_data['event'] = 'completed'
        self.publisher.send_progress(final_data)
        
        self.publisher.close()

# Main PC (Monitor) Script
class ProgressMonitor:
    """Monitor that runs on main PC"""
    def __init__(self):
        self.pc_info = get_pcinfo()
        port = get_network_info()["remote_camera"]
        self.pc_list = list(self.pc_info.keys())
        self.socket_dict = {pc_name:get_client_socket(self.pc_info[pc_name]["ip"], port) for pc_name in self.pc_list}
        
        self.progress_data = {pc_name: {
            'status': 'waiting',
            'progress_percent': 0,
            'completed': False,
            'last_update': 'Never'
        } for pc_name in self.pc_list}
                
        self.app = Flask(__name__)
        self.setup_web()
        
    def register(self):
        self.send_message("register")   
        return self.wait_for_message("registered")
    
    def send_message(self, message):
        for pc_name, socket in self.socket_dict.items():
            socket.send_string(message)
    
    def wait_for_message(self, message, timeout=-1):
        recv_dict = {pc_name:False for pc_name in self.pc_list}
        start_time = time.time()
        while timeout == -1 or time.time()-start_time < timeout:
            success = True
            for pc_name, socket in self.socket_dict.items():
                if recv_dict[pc_name]:
                    continue
                recv_msg = socket.recv_string()
                if recv_msg == message:
                    recv_dict[pc_name] = True

                if not recv_dict[pc_name]:
                    success = False
            if success:
                return True                
            time.sleep(0.01)
            
        return False
    
    def setup_web(self):
        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE, 
                                        pc_data=self.progress_data,
                                        timestamp=datetime.now().strftime('%H:%M:%S'))
        
        @self.app.route('/api/data')
        def api_data():
            return {
                'pc_data': self.progress_data,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
    
    def start_web_server(self):
        """웹 서버를 별도 스레드에서 시작"""
        def run_server():
            self.app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
        
        web_thread = threading.Thread(target=run_server)
        web_thread.daemon = True
        web_thread.start()
        
        # 브라우저 자동 열기
        time.sleep(1)
        webbrowser.open('http://localhost:8080')
        print("Web monitor started at http://localhost:8080")
            
    def monitor(self):
        """Start monitoring progress"""
        self.register()
        self.start_web_server()
        self.end_dict = {pc_name:False for pc_name in self.socket_dict.keys()}
        
        try:
            while True:
                all_completed = True
                for pc_name, socket in self.socket_dict.items():
                    try:
                        # Receive progress update from each PC
                        message = socket.recv_string(zmq.NOBLOCK)
                        data = json.loads(message)
                        
                        if data.get('event') == 'end':
                            self.progress_data[pc_name] = {
                                'status': 'COMPLETED',
                                'progress_percent': 100,
                                'completed': True,
                                'last_update': datetime.now().strftime('%H:%M:%S'),
                                'overall_progress': data.get('overall_progress', {}),
                                'video_progress': data.get('video_progress', {}),
                                'log_messages': data.get('log_messages', []),
                                'active_videos': data.get('active_videos', 0),
                                'save_path': data.get('save_path', '')
                            }
                            self.end_dict[pc_name] = True
                        else:
                            # 모든 진행률 데이터 업데이트
                            overall = data.get('overall_progress', {})
                            self.progress_data[pc_name] = {
                                'status': data.get('status', 'processing'),
                                'progress_percent': overall.get('progress_percent', 0),
                                'completed': False,
                                'last_update': datetime.now().strftime('%H:%M:%S'),
                                'overall_progress': overall,
                                'video_progress': data.get('video_progress', {}),
                                'log_messages': data.get('log_messages', []),
                                'active_videos': data.get('active_videos', 0),
                                'save_path': data.get('save_path', '')
                            }
                        
                        if not self.end_dict[pc_name]:
                            all_completed = False
                            
                    except zmq.Again:
                        # No message from this PC
                        if not self.end_dict[pc_name]:
                            all_completed = False
                        continue
                
                if all_completed:
                    print("\n=== ALL PCs COMPLETED ===")
                    break
                    
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"Error in monitoring: {e}")
        finally:
            self.close()
    
    def _display_progress(self, data):
        """Display progress information"""
        os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
        
        print("=" * 60)
        print(f"VIDEO PROCESSING MONITOR - {data['timestamp']}")
        print("=" * 60)
        
        overall = data['overall_progress']
        print(f"Status: {data['status'].upper()}")
        print(f"Overall Progress: {overall['progress_percent']:.1f}% ({overall['processed_frames']}/{overall['total_frames']} frames)")
        print(f"Processing Speed: {overall['fps']:.1f} FPS")
        print(f"Elapsed Time: {overall['elapsed_time']:.1f}s")
        print(f"ETA: {overall['eta_seconds']:.1f}s")
        print(f"Active Videos: {data['active_videos']}")
        
        # Progress bar
        progress = overall['progress_percent']
        bar_length = 50
        filled = int(bar_length * progress / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"[{bar}] {progress:.1f}%")
        
        print("\nPER-VIDEO PROGRESS:")
        print("-" * 40)
        for video_name, video_data in data['video_progress'].items():
            print(f"{video_name[:30]:30s} {video_data['progress_percent']:6.1f}% ({video_data['processed_frames']}/{video_data['total_frames']})")
        
        if data.get('log_messages'):
            print(f"\nRECENT LOGS:")
            print("-" * 20)
            for log in data['log_messages'][-5:]:  # Show last 5 logs
                print(f"  {log}")
        
        if data.get('event'):
            print(f"\nEVENT: {data['event']}")
        
        if data.get('error'):
            print(f"ERROR: {data['error']}")
    
    def close(self):
        for socket in self.socket_dict.values():
            socket.close()
            del socket

# HTML 템플릿

# HTML 템플릿
# HTML 템플릿
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Video Processing Monitor</title>
    <meta http-equiv="refresh" content="3">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: white; text-align: center; margin-bottom: 5px; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .subtitle { text-align: center; color: rgba(255,255,255,0.8); margin-bottom: 30px; font-size: 1.1em; }
        .pc-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(700px, 1fr)); gap: 20px; }
        
        .pc-card { 
            background: rgba(255,255,255,0.95); 
            border-radius: 15px; 
            padding: 25px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .pc-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .pc-name { font-weight: bold; font-size: 1.4em; color: #333; }
        .status { 
            padding: 8px 16px; border-radius: 20px; font-size: 12px; font-weight: bold;
            text-transform: uppercase; letter-spacing: 1px;
        }
        .status.waiting { background: #ffc107; color: #000; }
        .status.processing { background: #2196F3; color: white; }
        .status.completed { background: #4CAF50; color: white; }
        
        .overall-progress { margin: 20px 0; }
        .progress-label { display: flex; justify-content: space-between; margin-bottom: 5px; font-weight: 600; color: #555; }
        .progress-bar { 
            width: 100%; height: 25px; background: #e0e0e0; border-radius: 15px; 
            overflow: hidden; position: relative; margin-bottom: 15px;
        }
        .progress-fill { 
            height: 100%; background: linear-gradient(90deg, #4CAF50, #45a049); 
            transition: width 0.5s ease; border-radius: 15px;
            position: relative;
        }
        .progress-text { 
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
            color: white; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
        }
        
        .stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin: 20px 0; }
        .stat-item { 
            background: #f8f9fa; padding: 12px; border-radius: 8px; text-align: center;
            border-left: 4px solid #4CAF50;
        }
        .stat-value { font-size: 1.3em; font-weight: bold; color: #333; }
        .stat-label { font-size: 0.9em; color: #666; margin-top: 4px; }
        
        .videos-section { margin-top: 25px; }
        .videos-title { 
            font-size: 1.1em; font-weight: bold; color: #333; margin-bottom: 15px;
            border-bottom: 2px solid #e0e0e0; padding-bottom: 8px;
        }
        .video-item { 
            display: flex; align-items: center;
            padding: 8px 0; border-bottom: 1px solid #f0f0f0;
            gap: 15px;
        }
        .video-item:last-child { border-bottom: none; }
        .video-name { 
            flex: 1; font-weight: 500; color: #444; 
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .video-progress { 
            display: flex; align-items: center; gap: 10px;
            min-width: 150px;
        }
        .video-percent { 
            min-width: 50px; text-align: right; font-size: 0.9em; font-weight: 600; 
        }
        .video-bar { 
            width: 100px; height: 12px; background: #e0e0e0; border-radius: 6px; 
            overflow: hidden;
        }
        .video-fill { height: 100%; background: #2196F3; border-radius: 4px; transition: width 0.3s ease; }
        
        .logs-section { 
            margin-top: 20px; background: #f8f9fa; border-radius: 8px; padding: 15px;
            max-height: 150px; overflow-y: auto;
        }
        .logs-title { font-weight: bold; margin-bottom: 10px; color: #555; }
        .log-item { 
            font-size: 0.85em; color: #666; margin: 4px 0; 
            font-family: 'Courier New', monospace;
        }
        
        .no-data { text-align: center; color: #888; font-style: italic; padding: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 Video Processing Monitor</h1>
        <div class="subtitle">Real-time Multi-PC Processing Dashboard • Last Update: {{ timestamp }}</div>
        
        <div class="pc-grid">
            {% for pc_name, data in pc_data.items() %}
            <div class="pc-card">
                <div class="pc-header">
                    <div class="pc-name">📱 {{ pc_name }}</div>
                    <div class="status {{ data.status.lower() }}">{{ data.status }}</div>
                </div>
                
                <div class="overall-progress">
                    <div class="progress-label">
                        <span>Overall Progress</span>
                        <span>{{ "%.1f"|format(data.get('progress_percent', 0)) }}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {{ data.get('progress_percent', 0) }}%">
                            <div class="progress-text">{{ "%.1f"|format(data.get('progress_percent', 0)) }}%</div>
                        </div>
                    </div>
                </div>
                
                {% if data.get('overall_progress') %}
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value">{{ data.overall_progress.processed_frames }}</div>
                        <div class="stat-label">Processed</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{{ data.overall_progress.total_frames }}</div>
                        <div class="stat-label">Total Frames</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{{ "%.1f"|format(data.overall_progress.fps) }}</div>
                        <div class="stat-label">FPS</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{{ "%.0f"|format(data.overall_progress.eta_seconds) }}s</div>
                        <div class="stat-label">ETA</div>
                    </div>
                </div>
                {% endif %}
                
                {% if data.get('video_progress') %}
                <div class="videos-section">
                    <div class="videos-title">📹 Individual Videos ({{ data.video_progress|length }})</div>
                    {% for video_name, video_data in data.video_progress.items() %}
                    <div class="video-item">
                        <div class="video-name" title="{{ video_name }}">{{ video_name }}</div>
                        <div class="video-progress">
                            <div class="video-percent">{{ "%.1f"|format(video_data.progress_percent) }}%</div>
                            <div class="video-bar">
                                <div class="video-fill" style="width: {{ video_data.progress_percent }}%"></div>
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
                
                {% if data.get('log_messages') %}
                <div class="logs-section">
                    <div class="logs-title">📝 Recent Logs</div>
                    {% for log in data.log_messages[-5:] %}
                    <div class="log-item">{{ log }}</div>
                    {% endfor %}
                </div>
                {% endif %}
                
                {% if not data.get('overall_progress') %}
                <div class="no-data">No processing data available</div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
'''