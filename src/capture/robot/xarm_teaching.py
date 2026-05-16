import os
import time
from threading import Event
import argparse

from xarm.wrapper import XArmAPI
import cv2
import numpy as np

from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.system import network_info
from paradex.transforms.conversion import aa2mtx
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.io.capture_pc.command_sender import CommandSender
from paradex.io.capture_pc.ssh import run_script
from paradex.image.merge import merge_image

stop_event = Event()
save_event = Event()
listen_keyboard({'q':stop_event, 'c':save_event})

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='system/current/hecalib/xarm')
parser.add_argument('--stream', action='store_true',
                    help='Show live merged stream from remote capture PCs.')
args = parser.parse_args()

if args.save_path is not None:
    os.makedirs(args.save_path, exist_ok=True)

arm = XArmAPI(network_info["xarm"]["param"]["ip"], is_radian=True)
if arm.has_err_warn:
    arm.clean_error()

arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

# Turn on manual mode before recording
arm.set_mode(2)
arm.set_state(0)

rcc = None
dc = None
cs = None
img_dict = {}
img_text = {}
if args.stream:
    print("[stream] launching stream_client on capture PCs...")
    run_script("python src/capture/camera/stream_client.py")
    print("[stream] constructing remote_camera_controller (pinging daemons)...")
    rcc = remote_camera_controller("xarm_teaching")
    print("[stream] starting DataCollector (port 1234) ...")
    dc = DataCollector()
    dc.start()
    print("[stream] starting CommandSender (port 6890) ...")
    cs = CommandSender()
    print("[stream] sending start('stream') to daemons ...")
    rcc.start("stream", False, fps=30)
    print("[stream] daemons started. Entering view loop.")
    cv2.namedWindow("Teaching Stream", cv2.WINDOW_NORMAL)
    blank = np.ones((600, 800, 3), dtype=np.uint8) * 200
    cv2.putText(blank, "Waiting for stream...", (50, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("Teaching Stream", blank)
    cv2.waitKey(1)

idx = 0
try:
    while not stop_event.is_set():
        if args.stream:
            all_data = dc.get_data()
            for item_name, item_data in all_data.items():
                if item_data.get('type') != 'image':
                    continue
                image_bytes = item_data.get('data')
                frame_id = item_data.get('frame_id', 0)
                if not image_bytes:
                    continue
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is not None:
                    img_dict[item_name] = image
                    img_text[item_name] = str(frame_id)

            if img_dict:
                merged = merge_image(img_dict, img_text)
                cv2.imshow("Teaching Stream",
                           cv2.resize(merged, dsize=None, fx=0.8, fy=0.8))
            else:
                blank = np.ones((600, 800, 3), dtype=np.uint8) * 200
                cv2.putText(blank, "Waiting for stream...", (50, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.imshow("Teaching Stream", blank)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
            elif key == ord('c'):
                save_event.set()

        if save_event.is_set() and not args.save_path is None:
            _, pos_aa = arm.get_position_aa(is_radian=True)
            _, qpos = arm.get_joint_states()
            qpos = qpos[0][:6]

            pos_aa = np.array(pos_aa)
            wrist_pos = aa2mtx(pos_aa)

            np.save(os.path.join(args.save_path, f'{idx}_qpos.npy'), qpos)
            np.save(os.path.join(args.save_path, f'{idx}_aa.npy'), wrist_pos)

            print(f"Saved pose {idx}: {wrist_pos}")
            idx += 1
            save_event.clear()

        if not args.stream:
            time.sleep(0.1)

except KeyboardInterrupt:
    print("Interrupted by user.")

if args.stream:
    rcc.stop()
    rcc.end()
    dc.end()
    cs.end()
    cv2.destroyAllWindows()

# Turn off manual mode after recording
arm.set_mode(0)
arm.set_state(0)
arm.motion_enable(enable=False)
arm.disconnect()
print("Recording session ended.")
