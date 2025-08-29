import zmq
import logging
import json
import os
import threading
import time
from multiprocessing import Process, Queue, Manager

from paradex.io.capture_pc.util import get_client_socket, get_server_socket
from paradex.utils.env import get_pcinfo, get_network_info
from paradex.io.capture_pc.util import get_client_socket, get_server_socket
from paradex.utils.file_io import shared_dir

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
                with self.log_list.get_lock():
                    logs = list(self.log_list)
                    self.log_list[:] = []  
                self.log_socket.send_string(json.dumps(logs))
                
    def listen(self):
        while True:
            ident, msg = self.socket.recv_multipart()
            msg = msg.decode()
            print(msg)
            
            if msg == "quit":
                break
            
            if "start" in msg:
                file_name = msg.split(":")[1]
                root_path = os.path.join(shared_dir, file_name)
                
                process = Process(target=self._run_process_with_logging, args=(root_path))
                process.start()
        
        self.send_message("terminated")
        self.finished = True
                        
    def _run_process_with_logging(self, root_path):
        try:
            self.process(root_path, self.log_list)
            self.log_list.append({"root_dir":root_path, "time":time.time(), "state":"success", "msg":"", "type":"state"})
        except Exception as e:
            self.log_list.append({"root_dir":root_path, "time":time.time(), "state":"error", "msg":str(e), "type":"state"})    

class ProcessorMain():
    def __init__(self, process_dir_list):
        self.pc_info = get_pcinfo()
        net_info = get_network_info()
        port = net_info["remote_command"]
        log_port = net_info["remote_data"]
        self.pc_list = list(self.pc_info.keys())
        
        self.socket_dict = {pc_name:get_client_socket(self.pc_info[pc_name]["ip"], port) for pc_name in self.pc_list}
        self.log_socket = get_server_socket(log_port)
        
        self.process_dir_list = process_dir_list.copy()
        self.pc_state = {pc_name: "idle" for pc_name in self.pc_list}  # idle, processing
        self.current_tasks = {}  # pc_name -> current_dir
        self.finish = False
        
        monitor_thread = threading.Thread(target=self.monitor, daemon=True)
        monitor_thread.start()
        
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
                log_data = self.log_socket.recv_string(zmq.NOBLOCK)
                logs = json.loads(log_data)
                
                # 로그 처리 (완료 상태 확인)
                for log in logs:
                    print(log)
                    if log.get("state") == "success" or log.get("state") == "error":
                        pc_name = log.get("pc")
                        if pc_name in self.pc_state:
                            self.pc_state[pc_name] = "idle"  # 작업 완료
                            
            except zmq.Again:
                time.sleep(0.1)
            except Exception as e:
                print(f"Log receive error: {e}")
           
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
                    
                    # 작업 전송
                    message = f"start:{next_dir}"
                    self.socket_dict[pc_name].send_string(message)
                    print(pc_name, message)
            
            # 모든 작업 완료 시 종료
            if not self.process_dir_list and all(state == "idle" for state in self.pc_state.values()):
                break
                
            time.sleep(1)  # 1초마다 체크
            
        self.send_message("quit")
        self.wait_for_message("terminated")
        self.finish = True