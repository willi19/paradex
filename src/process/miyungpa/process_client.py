# process_client.py (local PC에서 실행)

import zmq
import traceback
from .process import process_demo  # 실제 처리 함수

def client():
    ctx = zmq.Context()
    
    # Main PC에 연결
    socket = ctx.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    
    print("Client ready, waiting for tasks...")
    
    while True:
        # Main PC로부터 명령 받기
        msg = socket.recv_string()
        
        if msg == "quit":
            socket.send_string("bye")
            break
        
        if msg.startswith("process:"):
            demo_path = msg.replace("process:", "")
            
            try:
                print(f"Processing: {demo_path}")
                socket.send_string(f"started:{demo_path}")
                
                # 실제 처리
                process_demo(demo_path)
                
                print(f"Done: {demo_path}")
                socket.send_string(f"done:{demo_path}")
                
            except Exception as e:
                print(f"Error: {demo_path} - {e}")
                traceback.print_exc()
                socket.send_string(f"error:{demo_path}:{str(e)}")
    
    socket.close()
    ctx.term()

if __name__ == "__main__":
    client()