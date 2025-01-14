import socket
import json

def main():
    server_ip = "0.0.0.0"  # Listen on all interfaces
    server_port = 5000
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((server_ip, server_port))
        server_socket.listen()
        print(f"Server listening on {server_ip}:{server_port}")
        
        while True:
            conn, addr = server_socket.accept()
            with conn:
                print(f"Connection from {addr}")
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    try:
                        message = json.loads(data.decode('utf-8'))
                        print(f"Received data: {message}")
                    except json.JSONDecodeError:
                        print("Failed to decode JSON.")

if __name__ == "__main__":
    main()
