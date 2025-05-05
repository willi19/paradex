# server_command.py
import zmq
import threading
import os
import sys
from paradex.io.capture_pc.connect import reset_and_run
from paradex.utils.file_io import home_path
import time

context = zmq.Context()
socket = context.socket(zmq.ROUTER)  # ROUTER로 다중 클라이언트 대응
socket.bind("tcp://*:5556")

time.sleep(1)  # 서버가 바인딩될 때까지 대기
# 클라이언트 ID 저장
client_ids = set()
expected_clients = set()  # pc.json 기반으로 채워짐

# 필요한 PC 목록 추출
import json
pc_info_path = os.path.join(home_path, "paradex", "config", "environment", "pc.json")
with open(pc_info_path, 'r') as f:
    pc_info = json.load(f)
    expected_clients = set(pc_info.keys())

reset_and_run("src/io_test/client.py")
print("[Server] Waiting for all clients to register...")

def send_commands():
    while True:
        key = sys.stdin.read(1)
        if key == 'c':
            print("[Server] Sending command to all clients...")
            for ident, name in client_ids:
                socket.send_multipart([ident, b"", b"do_something"])

def receive_replies():
    while True:
        ident, _, reply = socket.recv_multipart()
        for id_, name in client_ids:
            if id_ == ident:
                print(f"[{name}] Replied: {reply.decode()}")
                break

threading.Thread(target=send_commands, daemon=True).start()
receive_replies()  # main thread는 계속 수신 담당
