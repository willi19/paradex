# client.py
import zmq
import socket

SERVER_IP = "192.168.0.2"
context = zmq.Context()
socket = context.socket(zmq.DEALER)
socket.identity = socket.gethostname().encode()
socket.connect(f"tcp://{SERVER_IP}:5556")

# 등록 메시지 전송
socket.send_string(f"register:{socket.identity.decode()}")

# ack 수신
ack = socket.recv_string()
print(f"[Client] Registration ack received")

# 명령 루프
while True:
    msg = socket.recv_string()
    if msg == "do_something":
        print("[Client] Trigger received.")
        # TODO: 실제 동작 수행
        socket.send_string("done")
