import zmq
import logging
import json
import os
import threading
import time
from multiprocessing import Process, Queue, Manager
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

from paradex.io.capture_pc.util import get_client_socket, get_server_socket
from paradex.utils.env import get_pcinfo, get_network_info
from paradex.io.capture_pc.util import get_client_socket, get_server_socket
from paradex.utils.file_io import shared_dir, pc_name

class ProcessorLocal():
    def __init__(self, process):
        self.process = process
        port = get_network_info()["remote_command"]
        self.socket = get_server_socket(port)
        manager = Manager()
        self.log_list = manager.list()
        self.finished = False
        
        self.register()
        
        log_thread = threading.Thread(target=self.send_log, daemon=True)
        log_thread.start()
        
        listen_thread = threading.Thread(target=self.listen, daemon=True)
        listen_thread.start()
        
    def send_message(self, message):
        if isinstance(message, dict):
            message = json.dumps(message)
    
        self.socket.send_multipart([self.ident, message.encode('utf-8')])
        
    def register(self):
        print("start register")
        ident, msg = self.socket.recv_multipart()
        main_pc_ip = ident.decode().split(":")[0]
        msg = msg.decode()
        if msg == "register":
            self.ident = ident
            log_port = get_network_info()["remote_data"]
            self.log_socket = get_client_socket(main_pc_ip, log_port)
            
        self.send_message("registered")  
    
    def send_log(self):
        while True:
            time.sleep(1)
            if self.log_list:
                logs = list(self.log_list)
                self.log_list[:] = []
                
                for i in range(len(logs)):
                    logs[i]["pc"] = pc_name
                    logs[i]['root_dir'] = logs[i]['root_dir'].replace(shared_dir, "")
                    if logs[i]['root_dir'][0] == "/":
                        logs[i]['root_dir'] = logs[i]['root_dir'][1:]
                      
                self.log_socket.send_string(json.dumps(logs))
                
    def listen(self):
        while True:
            ident, msg = self.socket.recv_multipart()
            msg = msg.decode()
            
            if msg == "quit":
                break
            
            if "start" in msg:
                file_name = msg.split(":")[1]
                root_path = os.path.join(shared_dir, file_name)
                process = Process(target=self._run_process_with_logging, args=(root_path, ))
                process.start()
        
        self.send_message("terminated")
        self.finished = True
                        
    def _run_process_with_logging(self, root_path):
        try:
            self.process(root_path, self.log_list)
            self.log_list.append({"root_dir":root_path, "time":time.time(), "state":"success", "msg":"", "type":"state"})
        except Exception as e:
            self.log_list.append({"root_dir":root_path, "time":time.time(), "state":"error", "msg":str(e), "type":"state"})    

class WebHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = self.server.processor.get_status_html()
            self.wfile.write(html.encode())
        
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            status = self.server.processor.get_status_json()
            self.wfile.write(json.dumps(status).encode())
        
        else:
            self.send_response(404)
            self.end_headers()

class ProcessorMain():
    def __init__(self, process_dir_list, web_port=8080):
        self.pc_info = get_pcinfo()
        net_info = get_network_info()
        port = net_info["remote_command"]
        log_port = net_info["remote_data"]
        self.pc_list = list(self.pc_info.keys())
        
        self.socket_dict = {pc_name:get_client_socket(self.pc_info[pc_name]["ip"], port) for pc_name in self.pc_list}
        self.log_socket = get_server_socket(log_port)
        
        self.process_dir_list = process_dir_list.copy()
        self.original_dir_list = process_dir_list.copy()  # 원본 보관
        self.pc_state = {pc_name: "idle" for pc_name in self.pc_list}
        self.current_tasks = {}  # pc_name -> current_dir
        self.task_status = {}   # root_dir -> {'pc': pc_name, 'state': state, 'start_time': time, 'logs': []}
        self.finish = False
        
        # 각 root_dir 초기 상태 설정
        for root_dir in self.original_dir_list:
            self.task_status[root_dir] = {
                'pc': '',
                'state': 'pending',
                'start_time': None,
                'end_time': None,
                'logs': [],
                'last_update': time.time()
            }
        
        # 웹 서버 시작
        self.start_web_server(web_port)
        
        monitor_thread = threading.Thread(target=self.monitor, daemon=True)
        monitor_thread.start()
        
    def start_web_server(self, port):
        """웹 서버 시작"""
        httpd = HTTPServer(('0.0.0.0', port), WebHandler)
        httpd.processor = self  # ProcessorMain 인스턴스를 서버에 연결
        
        web_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        web_thread.start()
        
        print(f"Web monitor started at http://localhost:{port}")
        
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
    
    def receive_logs(self):
        while True:
            try:
                ident, log_data = self.log_socket.recv_multipart(zmq.NOBLOCK)
                logs = json.loads(log_data)
                print(logs)
                # 로그 처리
                for log in logs:
                    root_dir = log.get("root_dir")
                    pc_name = log.get("pc")
                    
                    if root_dir in self.task_status:
                        # 로그 추가
                        self.task_status[root_dir]['logs'].append(log)
                        self.task_status[root_dir]['last_update'] = time.time()
                        
                        # 상태 업데이트
                        if log.get("state") == "success":
                            self.task_status[root_dir]['state'] = 'completed'
                            self.task_status[root_dir]['end_time'] = log.get('time')
                            if pc_name in self.pc_state:
                                self.pc_state[pc_name] = "idle"
                                
                        elif log.get("state") == "error":
                            self.task_status[root_dir]['state'] = 'error'
                            self.task_status[root_dir]['end_time'] = log.get('time')
                            if pc_name in self.pc_state:
                                self.pc_state[pc_name] = "idle"
                                
                        elif log.get("type") == "process_msg":
                            self.task_status[root_dir]['state'] = 'processing'
                            
            except zmq.Again:
                time.sleep(0.1)
           
    def monitor(self):
        self.register()
        
        log_thread = threading.Thread(target=self.receive_logs, daemon=True)
        log_thread.start()
        
        while True:
            # 각 PC 상태 확인하고 작업 할당
            for pc_name in self.pc_list:
                if self.pc_state[pc_name] == "idle" and self.process_dir_list:
                    # 남은 작업이 있고 PC가 놀고 있으면 작업 할당
                    next_dir = self.process_dir_list.pop(0)
                    self.pc_state[pc_name] = "processing"
                    self.current_tasks[pc_name] = next_dir
                    
                    # 작업 상태 업데이트
                    if next_dir in self.task_status:
                        self.task_status[next_dir]['pc'] = pc_name
                        self.task_status[next_dir]['state'] = 'assigned'
                        self.task_status[next_dir]['start_time'] = time.time()
                    
                    # 작업 전송
                    message = f"start:{next_dir}"
                    self.socket_dict[pc_name].send_string(message)
            
            # 모든 작업 완료 시 종료
            if not self.process_dir_list and all(state == "idle" for state in self.pc_state.values()):
                break
                
            time.sleep(1)  # 1초마다 체크
            
        self.send_message("quit")
        self.wait_for_message("terminated")
        self.finish = True
    
    def get_status_json(self):
        """JSON 형태로 상태 반환"""
        return {
            'pc_states': self.pc_state,
            'task_status': self.task_status,
            'pending_tasks': len(self.process_dir_list),
            'total_tasks': len(self.original_dir_list),
            'current_tasks': self.current_tasks,
            'timestamp': time.time()
        }
    
    def get_status_html(self):
        """HTML 형태로 상태 반환"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Processing Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 10px; margin-bottom: 20px; }
        .pc-status { display: flex; gap: 20px; margin-bottom: 20px; }
        .pc-card { border: 1px solid #ddd; padding: 10px; border-radius: 5px; min-width: 150px; }
        .pc-idle { background: #e8f5e8; }
        .pc-processing { background: #fff3cd; }
        .task-table { width: 100%; border-collapse: collapse; }
        .task-table th, .task-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .task-table th { background: #f0f0f0; }
        .state-pending { background: #f8f9fa; }
        .state-assigned { background: #fff3cd; }
        .state-processing { background: #d4edda; }
        .state-completed { background: #d1ecf1; }
        .state-error { background: #f8d7da; }
        .log-preview { max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .refresh-btn { padding: 5px 10px; margin-left: 10px; }
    </style>
    <script>
        function refreshPage() {
            location.reload();
        }
        setInterval(refreshPage, 5000); // 5초마다 자동 새로고침
    </script>
</head>
<body>
    <div class="header">
        <h1>Processing Monitor</h1>
        <p>Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        <button class="refresh-btn" onclick="refreshPage()">Refresh</button>
    </div>
    
    <h2>PC Status</h2>
    <div class="pc-status">
"""
        
        # PC 상태 표시
        for pc_name, state in self.pc_state.items():
            current_task = self.current_tasks.get(pc_name, "")
            css_class = f"pc-{state}"
            
            html += f"""
        <div class="pc-card {css_class}">
            <h3>{pc_name}</h3>
            <p>Status: {state}</p>
            <p>Task: {os.path.basename(current_task) if current_task else "None"}</p>
        </div>
"""
        
        html += """
    </div>
    
    <h2>Task Status</h2>
    <p>Pending: """ + str(len(self.process_dir_list)) + """ / Total: """ + str(len(self.original_dir_list)) + """</p>
    
    <table class="task-table">
        <tr>
            <th>Root Directory</th>
            <th>PC</th>
            <th>State</th>
            <th>Start Time</th>
            <th>Duration</th>
            <th>Latest Log</th>
        </tr>
"""
        
        # 작업 상태 표시
        for root_dir, status in self.task_status.items():
            state = status['state']
            pc = status['pc']
            start_time = status['start_time']
            end_time = status['end_time']
            logs = status['logs']
            
            # 시간 계산
            if start_time:
                if end_time:
                    duration = f"{end_time - start_time:.1f}s"
                else:
                    duration = f"{time.time() - start_time:.1f}s"
                start_time_str = datetime.fromtimestamp(start_time).strftime("%H:%M:%S")
            else:
                duration = "-"
                start_time_str = "-"
            
            # 최신 로그
            latest_log = ""
            if logs:
                last_log = logs[-1]
                if last_log.get('type') == 'process_msg':
                    latest_log = f"Processing: {last_log.get('msg', '')}"
                else:
                    latest_log = f"{last_log.get('state', '')}: {last_log.get('msg', '')}"
            
            css_class = f"state-{state}"
            
            html += f"""
        <tr class="{css_class}">
            <td>{os.path.basename(root_dir)}</td>
            <td>{pc}</td>
            <td>{state}</td>
            <td>{start_time_str}</td>
            <td>{duration}</td>
            <td class="log-preview" title="{latest_log}">{latest_log}</td>
        </tr>
"""
        
        html += """
    </table>
</body>
</html>
"""
        return html