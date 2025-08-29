import numpy as np
import zmq
import socket

def get_my_ip():
    # 방법 1: socket으로 외부 연결해서 내 IP 확인
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    my_ip = s.getsockname()[0]
    s.close()
    return my_ip

def get_server_socket(port): # This get the infermation
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(f"tcp://*:{port}")
    return socket

def get_client_socket(ip, port):
    context = zmq.Context()
    sock = context.socket(zmq.DEALER)
    ident = f"{get_my_ip()}:server".encode()
    sock.identity = ident
    sock.connect(f"tcp://{ip}:{port}")
    return sock

def register(socket):
    ident, msg = socket.recv_multipart()
    msg = msg.decode()
    if msg == "register":
        socket.send_multipart([ident, b"registered"])
        print(f"[Server] Client registered: {ident.decode()}")
        return ident
    else:
        return None
