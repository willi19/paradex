import datetime
import os
import argparse
from threading import Event
import time

from paradex.utils.file_io import find_latest_index
from paradex.utils.path import shared_dir
from paradex.dataset_acqusition.capture import CaptureSession
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.calibration.utils import save_current_camparam, save_current_C2R


parser = argparse.ArgumentParser()


parser.add_argument('--hand_side', type=str, default='right')
parser.add_argument('--name', type=str)
parser.add_argument('--capture_root', default="0814_human_hand")
parser.add_argument('--realsense', action='store_true')
parser.add_argument('--human_tactile', '--human-tactile', action='store_true')
parser.add_argument('--human_tactile_port', '--human-tactile-port', default="/dev/ttyACM0")
parser.add_argument('--human_tactile_baud_rate', '--human-tactile-baud-rate', type=int, default=115200)
parser.add_argument('--human_tactile_reset_wait', '--human-tactile-reset-wait', type=float, default=2.0)
parser.add_argument('--human_tactile_plot_refresh_interval', '--human-tactile-plot-refresh-interval', type=float, default=0.02)
parser.add_argument('--human_tactile_plot_max_samples', '--human-tactile-plot-max-samples', type=int, default=200)
preview_group = parser.add_mutually_exclusive_group()
preview_group.add_argument(
    '--camera-preview',
    dest='camera_preview',
    action='store_true',
    help='Show live capture-PC images from the port 5484 preview API in a separate thread.',
)
preview_group.add_argument(
    '--no-camera-preview',
    dest='camera_preview',
    action='store_false',
    help='Disable the independent capture-PC preview GUI (default).',
)
parser.set_defaults(camera_preview=False)
parser.add_argument('--camera-preview-port', type=int, default=5484)
parser.add_argument('--camera-preview-refresh-interval', type=float, default=0.2)
parser.add_argument('--camera-preview-request-timeout', type=float, default=1.5)

args = parser.parse_args()

stop_event = Event()
save_event = Event()
exit_event = Event()

listen_keyboard({"c": save_event, "q": exit_event, "s": stop_event})
cs = CaptureSession(
    camera=True,
    realsense=args.realsense,
    human_tactile=args.human_tactile,
    human_tactile_port=args.human_tactile_port,
    human_tactile_baud_rate=args.human_tactile_baud_rate,
    human_tactile_reset_wait=args.human_tactile_reset_wait,
    human_tactile_plot_realtime=args.human_tactile,
    human_tactile_plot_refresh_interval=args.human_tactile_plot_refresh_interval,
    human_tactile_plot_max_samples=args.human_tactile_plot_max_samples,
)
camera_preview = None
if args.camera_preview:
    from paradex.io.camera_system.capture_pc_preview import CapturePcPreviewGui
    from paradex.utils.system import get_pc_list

    camera_preview = CapturePcPreviewGui(
        pc_list=get_pc_list(),
        port=args.camera_preview_port,
        refresh_interval=args.camera_preview_refresh_interval,
        request_timeout=args.camera_preview_request_timeout,
    )
    camera_preview.start()


def refresh_guis():
    """Run every GUI event loop on the main thread."""

    if camera_preview is not None:
        camera_preview.show()
    if cs.human_tactile is not None:
        cs.human_tactile.refresh_plot()

name = args.name


# last_idx = int(find_latest_index(os.path.join(shared_dir, "capture", args.capture_root, args.hand_side, name)))
last_idx = int(find_latest_index(os.path.join(shared_dir, "capture", args.capture_root, args.hand_side, name)))

try:
    while not exit_event.is_set():
        refresh_guis()
        if not save_event.is_set():
            stop_event.clear()
            time.sleep(0.02)
            continue

        # index = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        last_idx += 1
        save_path = os.path.join("capture", args.capture_root, args.hand_side, name, str(last_idx))
        cs.start(save_path)
        # cs.start(os.path.join(name, str(last_idx)))

        print("Starting new recording session:", name)
        while not stop_event.is_set() and not exit_event.is_set():
            refresh_guis()
            time.sleep(0.02)

        cs.stop()

        save_current_C2R(os.path.join(shared_dir, save_path))

        print("Stopped recording session:", name)

        save_event.clear()
        stop_event.clear()

        # print("Press Enter")

        # paired_robot_episode = int(input(f"Enter the episode number of paired robot sequence for {args.name}: "))

        # paired_info_json_path = os.path.join(shared_dir, save_path, "paired_robot_episode.json")

        # with open(paired_info_json_path, "w") as f:
        #     json.dump(
        #         {
        #             "human hand episode": last_idx,
        #             "paired robot episode": paired_robot_episode,
        #         },
        #         f,
        #         indent=2,
        #     )

        print(f"============== episode {last_idx} done =========================")
finally:
    print("Exiting recording.")
    if camera_preview is not None:
        camera_preview.close()
    if getattr(cs, "save_path", None) is not None:
        cs.stop()
    cs.end()
