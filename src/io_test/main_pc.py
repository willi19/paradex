# server_command.py
import zmq
import threading
import os
import sys
from paradex.io.capture_pc.connect import reset_and_run
from paradex.utils.file_io import home_path

context = zmq.Context()
socket = context.socket(zmq.ROUTER)  # ROUTER로 다중 클라이언트 대응
socket.bind("tcp://*:5556")

# 클라이언트 ID 저장
client_ids = set()
expected_clients = set()  # pc.json 기반으로 채워짐

# 필요한 PC 목록 추출
import json
pc_info_path = os.path.join(home_path, "paradex", "config", "environment", "pc.json")
with open(pc_info_path, 'r') as f:
    pc_info = json.load(f)
    expected_clients = set(pc_info.keys())

# 1. Capture PC에서 client.py 실행시키기 (ZMQ client 준비)
reset_and_run("src/io_test/client.py")

print("[Server] Waiting for all clients to register...")

# 2. client 등록 수신
def listen_for_registration():
    while len(client_ids) < len(expected_clients):
        ident, _, msg = socket.recv_multipart()
        decoded = msg.decode()
        if decoded.startswith("register:"):
            name = decoded.split(":")[1]
            client_ids.add((ident, name))
            print(f"[Server] Registered {name}")
            socket.send_multipart([ident, b"", b"ack"])
    print("[Server] All clients registered.\nPress 'c' to broadcast.")

# 3. 명령 수신 및 전송
def command_loop():
    while True:
        key = sys.stdin.read(1)
        if key == 'c':
            print("[Server] Sending command to all clients...")
            for ident, name in client_ids:
                socket.send_multipart([ident, b"", b"do_something"])

            for ident, name in client_ids:
                _, _, reply = socket.recv_multipart()
                print(f"[{name}] Replied: {reply.decode()}")

threading.Thread(target=listen_for_registration, daemon=True).start()
command_loop()
