import zmq
import os
import threading

should_exit = False  # 공유 변수로 종료 제어
client_ident = None  # 메인 PC에서 온 ident 저장용

def listen_for_commands():
    global should_exit, client_ident
            
context = zmq.Context()
socket = context.socket(zmq.ROUTER)
socket.bind("tcp://*:5556")

threading.Thread(target=listen_for_commands, daemon=True).start()
while not should_exit:
    ident, msg = socket.recv_multipart()
    msg = msg.decode()

    if msg == "quit":
        print(f"[Server] Received quit from client")
        should_exit = True
        break

    else:
        client_ident = ident 
        socket.send_multipart([client_ident, msg.encode()])
    
