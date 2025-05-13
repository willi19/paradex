# client.py
import zmq
import socket as pysocket  # 충돌 방지를 위해 이름 변경

SERVER_IP = "192.168.0.2"

context = zmq.Context()
sock = context.socket(zmq.DEALER)
sock.identity = pysocket.gethostname().encode()
sock.connect(f"tcp://{SERVER_IP}:5556")

# 등록 메시지 전송
sock.send_string(f"register:{sock.identity.decode()}")

# # ack 수신
# ack = sock.recv_string()
# print(f"[Client] Registration ack received")

# 명령 루프
# for i in range(10):
while True:
    sock.send_string("done")
