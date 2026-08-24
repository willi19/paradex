import os
import argparse
import atexit
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from threading import Event

# import chime
# chime.theme('pokemon')

from paradex.dataset_acqusition.capture import CaptureSession
from paradex.utils.path import shared_dir
from paradex.utils.file_io import find_latest_index
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.system import get_pc_list
from paradex.io.robot_controller.inspire_f1_tactile_plotter import InspireF1RealtimeTactilePlotter
from paradex.io.streamdeck_pedal import MiddlePedalState

EXCLUDED_PCS = {}
camera_pc_list = [pc for pc in get_pc_list() if pc not in EXCLUDED_PCS]

parser = argparse.ArgumentParser()

parser.add_argument('--device', choices=['xsens', 'occulus', 'vive'], default="vive")
parser.add_argument('--hand-side', choices=['right', 'left', 'bimanual'], default="right")
vive_group = parser.add_mutually_exclusive_group()
vive_group.add_argument(
    '--use-vive', dest='use_vive', action='store_true',
    help='Use the VIVE wrist tracker for arm/wrist pose fusion (default).',
)
vive_group.add_argument(
    '--no-vive', dest='use_vive', action='store_false',
    help='Use Manus glove data only; suitable for hand-only teleoperation.',
)
parser.set_defaults(use_vive=True)
camera_group = parser.add_mutually_exclusive_group()
camera_group.add_argument(
    '--camera',
    dest='camera_mode',
    nargs='?',
    const='capture',
    choices=['capture', 'preview'],
    default='capture',
    help=(
        "Enable remote camera recording. Use '--camera=preview' to also show "
        'the independent live capture-PC preview window.'
    ),
)
camera_group.add_argument(
    '--no-camera',
    dest='camera_mode',
    action='store_const',
    const='off',
    help='Disable remote cameras, sync generator, and timestamp monitor.',
)
camera_group.add_argument(
    '--camera-preview',
    dest='camera_mode',
    action='store_const',
    const='preview',
    help="Alias for '--camera=preview'.",
)
parser.add_argument('--camera-preview-port', type=int, default=5484)
parser.add_argument('--camera-preview-refresh-interval', type=float, default=1.0 / 30.0)
parser.add_argument('--camera-preview-request-timeout', type=float, default=1.5)
parser.add_argument(
    '--no-timestamp',
    dest='timestamp',
    action='store_false',
    help='Disable the timestamp monitor connection.',
)
parser.set_defaults(timestamp=True)
parser.add_argument('--arm', type=str, default="xarm",
                    help="Arm controller name. Use 'none' (or empty) to disable arm control.")
parser.add_argument('--hand', type=str, default="inspire_f1",
                    help="Hand controller/retargetor name. Use 'none' for arm-only teleop; 'allegro_v5' for the direct-anchor retargeter; 'wuji' for optimization, 'wuji_direct' for direct mapping, or 'wuji_hybrid' for opt thumb + direct fingers.")
parser.add_argument('--capture_root', type=str, default="eccv2026/allegro_v5")
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--tactile', action="store_true")
parser.add_argument('--ip', action="store_true")
parser.add_argument(
    '--allegro-command-rate-hz',
    type=float,
    default=30.0,
    help=(
        'Maximum live target-update rate for Allegro V5 hands. Defaults to '
        '30 Hz, matching the Allegro alignment experiment; the driver holds '
        'the latest target between updates.'
    ),
)
parser.add_argument(
    '--allegro-teleop-log',
    nargs='?',
    const='auto',
    default=None,
    metavar='PATH',
    help=(
        'Record every commanded Allegro V5 tick with the MANUS frame/ergonomics, '
        'retargeter target, controller target, and measured ROS feedback. '
        'Without PATH, save a timestamped .npz below this capture name.'
    ),
)
parser.add_argument(
    '--inspire-right-interface',
    default='enp8s0f1',
    help='Network interface for the right Inspire Modbus TCP hand.',
)
parser.add_argument(
    '--inspire-right-ip',
    default='192.168.11.211',
    help='IP address for the right Inspire Modbus TCP hand.',
)
parser.add_argument(
    '--inspire-left-interface',
    default='enp8s0f2',
    help='Network interface for the left Inspire Modbus TCP hand.',
)
parser.add_argument(
    '--inspire-left-ip',
    default='192.168.11.210',
    help='IP address for the left Inspire Modbus TCP hand.',
)
parser.add_argument(
    '--visualize-tactile-realtime',
    action="store_true",
    help=(
        'Show live tactile feedback. Inspire F1 uses matplotlib; Allegro V5 '
        'uses a Viser feedback mesh with fingertip arrows.'
    ),
)
parser.add_argument(
    '--allegro-visualization-rate-hz',
    type=float,
    default=100.0,
    help='Viser refresh rate for lightweight Allegro tactile arrows.',
)
parser.add_argument(
    '--allegro-tactile-display-max',
    type=float,
    default=1000.0,
    help='Raw tactile magnitude represented by the maximum arrow length.',
)
parser.add_argument('--xarm-servo-api', choices=["cartesian_aa", "angle_j"], default="cartesian_aa")
parser.add_argument(
    '--scale', '--hand-scale',
    dest='hand_scale',
    type=float,
    default=1.15

    ,
    help="Uniform keypoint scale around the wrist for Wuji. Use 1.5 if the robot hand is about 1.5x larger.",
)
args = parser.parse_args()


def _normalize_optional_name(name):
    if name is not None and name.strip().lower() in ("", "none", "null"):
        return None
    return name


args.arm = _normalize_optional_name(args.arm)
args.hand = _normalize_optional_name(args.hand)
if args.allegro_command_rate_hz <= 0.0:
    parser.error('--allegro-command-rate-hz must be positive.')
if args.camera_preview_port <= 0:
    parser.error('--camera-preview-port must be positive.')
if args.camera_preview_refresh_interval <= 0.0:
    parser.error('--camera-preview-refresh-interval must be positive.')
if args.camera_preview_request_timeout <= 0.0:
    parser.error('--camera-preview-request-timeout must be positive.')
if args.allegro_visualization_rate_hz <= 0.0:
    parser.error('--allegro-visualization-rate-hz must be positive.')
if args.allegro_tactile_display_max <= 0.0:
    parser.error('--allegro-tactile-display-max must be positive.')

camera_enabled = args.camera_mode != 'off'
camera_preview_enabled = args.camera_mode == 'preview'
allegro_v5_hands = {'allegro_v5'}
allegro_realtime_visualization = (
    args.visualize_tactile_realtime
    and args.hand in allegro_v5_hands
    and args.hand_side == 'right'
)
allegro_teleop_diagnostic_path = None
if args.allegro_teleop_log is not None:
    if args.hand not in allegro_v5_hands or args.hand_side != 'right':
        parser.error('--allegro-teleop-log currently requires --hand allegro_v5 and --hand-side right.')
    if args.allegro_teleop_log == 'auto':
        diagnostic_name = datetime.now().strftime('allegro_teleop_%Y%m%d_%H%M%S.npz')
        allegro_teleop_diagnostic_path = (
            Path(shared_dir) / 'capture' / args.capture_root / args.name / diagnostic_name
        )
    else:
        allegro_teleop_diagnostic_path = Path(args.allegro_teleop_log).expanduser()


stop_event = Event()
save_event = Event()
exit_event = Event()
grasp_yes_event = Event()
grasp_no_event = Event()
events = {"save": save_event, "stop": stop_event, "exit": exit_event}

listen_keyboard(
    {
        "c": save_event,
        "q": exit_event,
        "s": stop_event,
        "y": grasp_yes_event,
        "n": grasp_no_event,
    }
)
print("Keyboard control: c=start capture, s=stop capture, q=exit, y=grasp success, n=grasp fail")
print("In keyboard_control mode, gesture states 2/3 do not control session start/stop/exit.")

pedal_state = MiddlePedalState() if args.hand_side == "bimanual" else None
if pedal_state is not None:
    atexit.register(pedal_state.close)

inspire_bimanual = (
    args.hand in ("inspire", "inspire_dftp")
    and args.hand_side == "bimanual"
)
hand_kwargs = None
if inspire_bimanual:
    hand_kwargs = {
        "right": {
            "interface": args.inspire_right_interface,
            "host": args.inspire_right_ip,
        },
        "left": {
            "interface": args.inspire_left_interface,
            "host": args.inspire_left_ip,
        },
    }
    print(
        "Bimanual Inspire Modbus TCP: "
        f"right={args.inspire_right_ip} via {args.inspire_right_interface}, "
        f"left={args.inspire_left_ip} via {args.inspire_left_interface}"
    )

try:
    cs = CaptureSession(
        camera=camera_enabled,
        realsense=False,
        arm=args.arm,
        hand=args.hand,
        teleop=args.device,
        hand_side=args.hand_side,
        events=events,
        tactile=args.tactile or allegro_realtime_visualization,
        ip=args.ip or inspire_bimanual,
        hand_kwargs=hand_kwargs,
        timestamp=args.timestamp,
        camera_pc_list=camera_pc_list,
        arm_kwargs={"servo_api": args.xarm_servo_api} if args.arm == "xarm" else None,
        hand_scale=args.hand_scale,
        hand_command_rate_hz=(
            args.allegro_command_rate_hz
            if args.hand == 'allegro_v5'
            else None
        ),
        allegro_teleop_diagnostic_path=allegro_teleop_diagnostic_path,
        use_vive=args.use_vive,
        require_left_control=args.use_vive,
    )
except Exception:
    if pedal_state is not None:
        pedal_state.close()
    raise

if allegro_teleop_diagnostic_path is not None:
    print(f"Allegro teleop diagnostic will be saved to: {allegro_teleop_diagnostic_path}")

bimanual_state_provider = pedal_state.get_state if pedal_state is not None else None

tactile_plotter = None
if args.visualize_tactile_realtime:
    if args.hand_side == "bimanual":
        print("Realtime tactile visualization is not supported in bimanual mode. Ignoring option.")
    elif args.hand in allegro_v5_hands and args.hand_side != "right":
        print("Realtime Allegro visualization currently supports the right hand only. Ignoring option.")
    elif args.hand in allegro_v5_hands:
        from paradex.visualization.allegro_realtime import AllegroRealtimeViser

        tactile_plotter = AllegroRealtimeViser(
            cs.hand,
            update_rate_hz=args.allegro_visualization_rate_hz,
            tactile_display_max=args.allegro_tactile_display_max,
        )
        tactile_plotter.start()
    elif args.hand != "inspire_f1":
        print("Realtime tactile visualization supports inspire_f1 and Allegro V5. Ignoring option.")
    elif not args.tactile:
        print("Realtime tactile visualization requires --tactile. Ignoring option.")
    else:
        tactile_plotter = InspireF1RealtimeTactilePlotter(cs.hand)
        if tactile_plotter.enabled:
            tactile_plotter.start()

camera_preview = None
if camera_preview_enabled:
    from paradex.io.camera_system.capture_pc_preview import CapturePcPreviewGui

    camera_preview = CapturePcPreviewGui(
        pc_list=camera_pc_list,
        port=args.camera_preview_port,
        refresh_interval=args.camera_preview_refresh_interval,
        request_timeout=args.camera_preview_request_timeout,
        side_panel_provider=getattr(tactile_plotter, "render_bgr", None),
    )
    camera_preview.start()


def refresh_guis(_session=None):
    """Process preview GUI events on the teleoperation main thread."""

    if camera_preview is not None:
        camera_preview.show()


def wait_for_grasp_result():
    grasp_yes_event.clear()
    grasp_no_event.clear()
    print("Grasp success? Press y or n.")

    while not exit_event.is_set():
        refresh_guis()
        if grasp_yes_event.is_set():
            return True
        if grasp_no_event.is_set():
            return False
        time.sleep(0.01)

    return None


name = args.name

last_idx = int(find_latest_index(os.path.join(shared_dir, "capture", args.capture_root, args.name)))
success_count = 0
fail_count = 0

try:
    while not exit_event.is_set():
        state = cs.teleop(
            session_events=events,
            state_policy="keyboard_control",
            loop_callback=refresh_guis,
            bimanual_state_provider=bimanual_state_provider,
        )

        if state == "exit":
            break

        if state != "start":
            continue

        last_idx += 1
        print("Prepare to record new session:", name, "episode:", last_idx)
        episode_rel_path = os.path.join("capture", args.capture_root, args.name, str(last_idx))
        episode_abs_path = os.path.join(shared_dir, episode_rel_path)
        # Match capture_hand.py: an idle-time 's' must not immediately stop
        # the next capture after 'c' is pressed.
        stop_event.clear()
        cs.start(episode_rel_path)
        # chime.info(sync=True)
        print("Starting new recording session:", name)
        print("Capturing index:", last_idx)

        state = cs.teleop(
            session_events=events,
            state_policy="keyboard_control",
            loop_callback=refresh_guis,
            bimanual_state_provider=bimanual_state_provider,
        )
        print("Stopping recording session:", name)
        cs.stop()
        print("Stopped recording session:", name)

        timestamp_npy_path = os.path.join(episode_abs_path, "raw", "timestamps", "timestamp.npy")
        if os.path.exists(timestamp_npy_path):
            print(f"timestamp.npy length: {len(np.load(timestamp_npy_path))}")
        else:
            print(f"timestamp.npy not found at {timestamp_npy_path}")

        save_event.clear()
        stop_event.clear()

        if state != "exit":
            grasp_success = wait_for_grasp_result()
            if grasp_success is not None:
                success_count += int(grasp_success)
                fail_count += int(not grasp_success)
                os.makedirs(episode_abs_path, exist_ok=True)
                grasp_json_path = os.path.join(episode_abs_path, "grasp_result.json")
                with open(grasp_json_path, "w") as file:
                    json.dump(
                        {
                            "episode": last_idx,
                            "grasp_success": grasp_success,
                        },
                        file,
                        indent=2,
                    )

                paired_human_episode = int(
                    input(
                        "Enter the episode number of paired human sequence "
                        f"for {args.name}: "
                    )
                )
                paired_info_json_path = os.path.join(
                    episode_abs_path,
                    "paired_human_episode.json",
                )
                with open(paired_info_json_path, "w") as file:
                    json.dump(
                        {
                            "human hand episode": last_idx,
                            "paired human episode": paired_human_episode,
                        },
                        file,
                        indent=2,
                    )

                print(
                    f"Current Success count: {success_count} / "
                    f"Failure count: {fail_count}"
                )

        grasp_yes_event.clear()
        grasp_no_event.clear()
        print(f"============== episode {last_idx} done =========================")

        if state == "exit":
            break
finally:
    print("Exiting teleoperation recording.")
    if camera_preview is not None:
        camera_preview.close()
    if tactile_plotter is not None:
        tactile_plotter.close()
    if getattr(cs, "save_path", None) is not None:
        cs.stop()
    cs.end()
    if pedal_state is not None:
        pedal_state.close()
