import zmq
import socket as pysocket

SERVER_IP = "192.168.0.2"

context = zmq.Context()
sock = context.socket(zmq.DEALER)
sock.identity = pysocket.gethostname().encode()
sock.connect(f"tcp://{SERVER_IP}:5556")

# 클라이언트 루프
while True:
    msg = sock.recv()  # ROUTER에서 보내는 건 단일 프레임
    msg = msg.decode()

    if msg == "capture":
        print("[Client] Received capture command")
        sock.send_string("received")  # ROUTER expects single-frame reply
