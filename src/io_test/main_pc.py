# register_server.py
import socket
import json

HOST = ""  # 모든 인터페이스에서 수신
PORT = 8888

def handle_connection(conn, addr):
    try:
        with conn:
            data = b""
            while True:
                chunk = conn.recv(1024)
                if not chunk:
                    break
                data += chunk
                if b"\n" in chunk:
                    break  # 하나의 메시지만 받음

            msg = json.loads(data.decode("utf-8").strip())
            if msg.get("type") == "register":
                print(f"[Server] Registered: {msg['hostname']} from {addr}")
            else:
                print(f"[Server] Unknown message: {msg}")
    except Exception as e:
        print(f"[Server] Error from {addr}: {e}")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"[Server] Listening on port {PORT}...")

    while True:
        conn, addr = s.accept()
        handle_connection(conn, addr)
