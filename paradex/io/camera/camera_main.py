import json
import zmq

class RemoteCameraController():
    def listen_socket(pc_name, socket):
        while True:
            msg = socket.recv_string()
            if msg == "terminated":
                terminate_dict[pc_name] = True
                print(f"[{pc_name}] Terminated.")
                break
            
            elif msg == "camera_ready":
                start_dict[pc_name] = True
                print(f"[{pc_name}] Camera ready.")
                continue
            
            elif msg == "camera_error":
                print(f"[{pc_name}] Camera error.")
                terminate_dict[pc_name] = True
                continue

            elif msg == "save_finish":
                capture_state[pc_name] = False
                print(f"[{pc_name}] Save finished.")
                continue

            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                print(f"[{pc_name}] Non-JSON message: {msg}")
                continue
            
            serial_num = data["serial_num"]
            if data.get("type") == "charuco":
                result = data["detect_result"]
                corners = np.array(result["checkerCorner"], dtype=np.float32)
                ids = np.array(result["checkerIDs"], dtype=np.int32).reshape(-1, 1)
                frame = data["frame"]
                cur_state[serial_num] = (corners, ids, frame)

                if result["save"]:
                    draw_charuco(saved_corner_img[serial_num], corners, BOARD_COLORS[2], 5, -1, ids)

            else:
                print(f"[{pc_name}] Unknown JSON type: {data.get('type')}")

context = zmq.Context()
socket_dict = {}
terminate_dict = {pc: False for pc in pc_info.keys()}
start_dict = {pc: False for pc in pc_info.keys()}
