# command_client.py
import zmq
import time

def run_command_listener(server_ip="127.0.0.1", port=5556):
    context = zmq.Context()

    # REQ socket: 명령 요청 대기
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{server_ip}:{port}")

    print(f"Connected to server at {server_ip}:{port}")

    while True:
        try:
            # 서버에게 "ready" 요청
            socket.send(b"ready")
            command = socket.recv_string()
            print(f"Received command: {command}")

            # 무조건 'yes'라고 응답
            socket.send_string("yes")
            ack = socket.recv_string()
            print(f"Acknowledged: {ack}")

            time.sleep(0.5)  # 너무 과도한 요청 방지

        except KeyboardInterrupt:
            print("Interrupted. Exiting.")
            break

    socket.close()
    context.term()
