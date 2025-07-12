
import argparse
import json
import numpy as np
import cv2
import os

from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.io.capture_pc.util import get_client_socket
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.image.aruco import draw_charuco
from paradex.utils.env import get_pcinfo

BOARD_COLORS = [
    (255, 0, 0),     # 빨강
    (0, 255, 0),     # 초록
    (0, 0, 255),     # 파랑
    (255, 255, 0),   # 시안
    (255, 0, 255),   # 마젠타
    (0, 255, 255),   # 노랑
    (128, 128, 255)  # 연보라
]

def get_pc_name(serial_num):
    pc_info = get_pcinfo()
    pc_name = None
    for pc in pc_info.keys():
        if serial_num in pc_info[pc]['cam_list']:
            pc_name = pc
            break

    if pc_name is None:
        raise ValueError(f"Serial number {serial_num} not found in PC list.")
    
    return pc_name, pc_info[pc_name]
            
parser = argparse.ArgumentParser(description="Capture intrinsic camera calibration.")
parser.add_argument(
    "--serial",
    type=str,
    required=True,
    help="Directory to save the video.",
)

args = parser.parse_args()
serial_num = args.serial

pc_name, pc_info = get_pc_name(serial_num)

git_pull("merging", [pc_name]) 
run_script(os.path.join(f"python src/calibration/intrinsic/client.py --serial {serial_num}"), [pc_name])  # 명령 수신 대기

camera_controller = RemoteCameraController("stream", [serial_num], sync=False)
camera_controller.start_capture()

socket = get_client_socket(pc_info["ip"], 5564)

saved_board_corners = []
saved_image = np.zeros((1536, 2048, 3), dtype=np.uint8)

try:
    while True:
        msg = socket.recv_string()
        data = json.loads(msg)
        plot_img = saved_image.copy()
        if data.get("type") == "charuco":
            for board_id, result in data["detect_result"].items():
                corners = np.array(result["checkerCorner"], dtype=np.float32)
                ids = np.array(result["checkerIDs"], dtype=np.int32).reshape(-1, 1)

                save = result["save"]
                if save:
                    saved_board_corners.append(corners)
                    draw_charuco(saved_image, corners, BOARD_COLORS[0], 5, -1, ids)

                if corners.shape[0] == 0:
                    continue
            
                draw_charuco(plot_img, corners, BOARD_COLORS[int(board_id)], 5, -1, ids)

            plot_img = cv2.resize(plot_img, (2048 // 2, 1536 // 2))
            cv2.imshow("Charuco Detection", plot_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                socket.send_string("quit")
                break
        else:
            print(f"[Client] Unknown JSON type: {data.get('type')}")

finally:
    camera_controller.end_capture()
    camera_controller.quit()        