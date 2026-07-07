"""Main-PC orchestrator for distributed intrinsic calibration.

Starts camera streams on every Capture PC, spawns intrinsic/client.py on each,
and shows a merged live preview. There is NO save button: each Capture PC
auto-accumulates novel charuco frames per camera (see client.py). This window
only shows the current detection (green) and each camera's kept-frame count
(`kept/target`), so you can walk the board around until every tile is full.

Press 'q' to finish. The Capture PCs then write their keypoint .npy files;
run calculate.py afterward to solve K + distortion per camera.

No `--args`. Coupled to client.py by data types and ports (1234 publish,
6890 command) — change both together.
"""

import cv2
import numpy as np

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.capture_pc.ssh import run_script
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.io.capture_pc.command_sender import CommandSender
from paradex.image.merge import merge_image
from paradex.image.aruco import draw_charuco

CUR_COLOR = (0, 255, 0)

run_script("python src/calibration/intrinsic/client.py")

rcc = remote_camera_controller("intrinsic_calibration")
dc = DataCollector()
dc.start()

cs = CommandSender()
rcc.arm(syncMode=False, fps=30)
rcc.set_stream(True)

img_dict = {}
img_text = {}
cur_state = {}  # serial_num -> current corners (downscaled /8)

while True:
    all_data = dc.get_data()
    for item_name, item_data in all_data.items():
        if item_data.get('type') == 'image':
            image_bytes = item_data.get('data')
            if image_bytes:
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is not None:
                    img_dict[item_name] = image
                    kept = item_data.get('kept', 0)
                    target = item_data.get('target', 0)
                    img_text[item_name] = f"{kept}/{target}"

        elif item_data.get('type') == 'charuco_detection':
            data = item_data.get('data')
            serial_num = item_name.split("_")[0]
            cur_state[serial_num] = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)

    if img_dict:
        display_dict = {}
        for serial_num in img_dict.keys():
            disp = img_dict[serial_num].copy()
            corners = cur_state.get(serial_num, np.zeros((0, 2), dtype=np.float32))
            if corners.shape[0] > 0:
                draw_charuco(disp, corners, CUR_COLOR, 1, -1)
            display_dict[serial_num] = disp

        merged_image = merge_image(display_dict, img_text)
        cv2.imshow("Intrinsic Capture", cv2.resize(merged_image, dsize=None, fx=0.8, fy=0.8))
        key = cv2.waitKey(1)
    else:
        blank_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.putText(blank_image, "Waiting for stream...", (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("Intrinsic Capture", blank_image)
        key = cv2.waitKey(1)

    if key == ord('q'):
        break

print("Stopping capture...")

rcc.stop()
rcc.end()
dc.end()
cs.end()  # sends 'exit' -> Capture PCs save their keypoint files

cv2.destroyAllWindows()
print("Stream stopped. Capture PCs are saving keypoints; run calculate.py next.")
