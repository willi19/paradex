# # register_client.py
# import socket
# import json
# import os

# MAIN_PC_IP = "192.168.0.1"  # ← Main PC의 실제 IP로 교체
# PORT = 5555

# message = {
#     "type": "register",
#     "hostname": os.uname()[1]  # 또는 socket.gethostname()
# }

# try:
#     with socket.create_connection((MAIN_PC_IP, PORT), timeout=5) as sock:
#         sock.sendall((json.dumps(message) + "\n").encode("utf-8"))
#         print("[Client] Sent registration message.")
# except Exception as e:
#     print(f"[Client] Connection failed: {e}")
