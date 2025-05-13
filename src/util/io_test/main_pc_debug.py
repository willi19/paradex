import zmq

ctx = zmq.Context()
sock = ctx.socket(zmq.ROUTER)
sock.bind("tcp://*:5556")

while True:
    msg = sock.recv_multipart()
    print(f"[ROUTER] Received: {msg}")
