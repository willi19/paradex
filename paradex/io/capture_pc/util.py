import numpy as np
import zmq

def get_server_socket(port): # This get the infermation
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(f"tcp://*:{port}")
    return socket

def get_client_socket(ip, port):
    context = zmq.Context()
    sock = context.socket(zmq.DEALER)
    sock.identity = b"server"
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
