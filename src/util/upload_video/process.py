"""
Main PC에서 실행: Worker PC들의 진행상황을 수집하고 웹으로 표시
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import time
from threading import Thread
import os

from paradex.io.capture_pc.data_sender import DataCollector
from paradex.io.capture_pc.ssh import run_script

class VideoProgressMonitor:
    """비디오 처리 진행상황 웹 모니터"""
    
    def __init__(self, web_port=8080, zmq_port=1234):
        self.web_port = web_port
        self.zmq_port = zmq_port
        
        # Flask 앱
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'video_monitor_secret'
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*",
            async_mode='threading'
        )
        
        # DataCollector 초기화
        self.collector = DataCollector(port=zmq_port)
        
        # 라우트 설정
        self.setup_routes()
        self.setup_socketio()
        
    def setup_routes(self):
        """Flask 라우트 설정"""
        
        @self.app.route('/')
        def index():
            return render_template('video_monitor.html')
        
        @self.app.route('/api/progress')
        def get_progress():
            """현재 진행상황 API"""
            data = self.collector.get_data()
            
            # 통계 계산
            total_videos = len(data)
            completed = sum(1 for v in data.values() if v.get('status') == 'completed')
            processing = sum(1 for v in data.values() if v.get('status') == 'processing')
            failed = sum(1 for v in data.values() if v.get('status') == 'failed')
            
            total_progress = sum(v.get('progress', 0) for v in data.values())
            avg_progress = total_progress / total_videos if total_videos > 0 else 0
            
            return jsonify({
                'videos': data,
                'summary': {
                    'total': total_videos,
                    'completed': completed,
                    'processing': processing,
                    'failed': failed,
                    'avg_progress': avg_progress
                }
            })
    
    def setup_socketio(self):
        """SocketIO 이벤트 설정"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
            # 초기 데이터 전송
            data = self.collector.get_data()
            self.socketio.emit('initial_data', {'videos': data})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('Client disconnected')
    
    def update_loop(self):
        """주기적으로 클라이언트에게 업데이트 전송"""
        last_data = {}
        
        while True:
            current_data = self.collector.get_data()
            
            # 변경된 데이터만 전송
            changed = {}
            for name, info in current_data.items():
                if name not in last_data or last_data[name] != info:
                    changed[name] = info
            
            if changed:
                # 통계 계산
                total_videos = len(current_data)
                completed = sum(1 for v in current_data.values() if v.get('status') == 'completed')
                processing = sum(1 for v in current_data.values() if v.get('status') == 'processing')
                failed = sum(1 for v in current_data.values() if v.get('status') == 'failed')
                
                total_progress = sum(v.get('progress', 0) for v in current_data.values())
                avg_progress = total_progress / total_videos if total_videos > 0 else 0
                
                self.socketio.emit('progress_update', {
                    'videos': changed,
                    'summary': {
                        'total': total_videos,
                        'completed': completed,
                        'processing': processing,
                        'failed': failed,
                        'avg_progress': avg_progress
                    }
                })
            
            last_data = current_data.copy()
            time.sleep(1.0)  # 1초마다 업데이트
    
    def start(self):
        """모니터 시작"""
        print(f"Starting Video Progress Monitor...")
        print(f"Web interface: http://localhost:{self.web_port}")
        print(f"ZMQ port: {self.zmq_port}")
        
        # Collector 시작
        self.collector.start()
        
        # 업데이트 스레드 시작
        update_thread = Thread(target=self.update_loop, daemon=True)
        update_thread.start()
        
        # Flask 서버 시작
        self.socketio.run(
            self.app,
            host='0.0.0.0',
            port=self.web_port,
            debug=False,
            use_reloader=False
        )


if __name__ == "__main__":
    run_script('python src/util/upload_video/client.py')
    monitor = VideoProgressMonitor(web_port=8081, zmq_port=1234)
    monitor.start()