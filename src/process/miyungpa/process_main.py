# processor.py (ProcessorMain 클래스)

import zmq
import time
from threading import Thread
from paradex.utils.system import get_pc_list, get_pc_ip

class ProcessorMain:
    def __init__(self, process_list):
        self.process_list = process_list.copy()
        self.pc_list = get_pc_list()
        self.finish = False
        
        # 각 PC에 소켓 연결
        self.ctx = zmq.Context()
        self.sockets = {}
        for pc_name in self.pc_list:
            socket = self.ctx.socket(zmq.REQ)
            socket.setsockopt(zmq.RCVTIMEO, 5000)
            socket.connect(f"tcp://{get_pc_ip(pc_name)}:5555")
            self.sockets[pc_name] = socket
        
        self.pc_busy = {pc: False for pc in self.pc_list}
        
        # 작업 분배 시작
        Thread(target=self._distribute, daemon=True).start()
    
    def _distribute(self):
        while self.process_list or any(self.pc_busy.values()):
            for pc_name in self.pc_list:
                if not self.pc_busy[pc_name] and self.process_list:
                    task = self.process_list.pop(0)
                    Thread(target=self._process_task, args=(pc_name, task), daemon=True).start()
            
            time.sleep(0.5)
        
        # 종료
        for pc_name in self.pc_list:
            try:
                self.sockets[pc_name].send_string("quit")
                self.sockets[pc_name].recv_string()
            except:
                pass
        
        self.finish = True
    
    def _process_task(self, pc_name, task):
        self.pc_busy[pc_name] = True
        socket = self.sockets[pc_name]
        
        try:
            # 작업 전송
            socket.send_string(f"process:{task}")
            
            # 시작 확인
            response = socket.recv_string()
            print(f"[{pc_name}] {response}")
            
            # 완료 대기
            socket.send_string("status")
            response = socket.recv_string()
            print(f"[{pc_name}] {response}")
            
        except Exception as e:
            print(f"[{pc_name}] Error: {e}")
        
        finally:
            self.pc_busy[pc_name] = False