# server_command.py
import zmq
from paradex.io.capture_pc.connect import reset_and_run
import os
from paradex.utils.file_io import home_path

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5556")

reset_and_run(os.path.join("src/io_test/client.py"))  # 명령 수신 대기

# while True:
#     msg = socket.recv_string()
#     print(f"Client says: {msg}")
#     socket.send_string("do_something")  # 명령 송신

#     reply = socket.recv_string()
#     print(f"Client replied: {reply}")
#     socket.send_string("ack")  # 수신 확인
