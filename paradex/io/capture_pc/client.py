import zmq

def get_socket(port=5556):
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(f"tcp://*:{port}")
    return socket

def register(socket):
    ident, msg = socket.recv_multipart()
    msg = msg.decode()
    if msg == "register":
        socket.send_multipart([ident, b"registered"])
        print(f"[Server] Client registered: {ident.decode()}")
        return ident
    else:
        return None