"""HRI teleop capture: VIVE wrist + MANUS gloves -> xArm + Inspire, with remote
cameras (free-run, no hardware sync generator).

Prerequisites (must be running / sourced in this shell):
  - VIVE ROS2 publisher    : vive-teleop/scripts/ros2_publish.sh  -> /vive_trackers/right_hand/pose
  - MANUS ROS2 publisher   : vive-teleop/scripts/manus_publish_remote.sh -> /manus_glove_0,1
  - ROS 2 sourced          : source /opt/ros/humble/setup.bash
                             source <vive-teleop>/ros2_ws/install/setup.bash   (manus_ros2_msgs)
  - Both MANUS gloves on   : right hand teleops the robot, left hand is the
                             gesture/clutch channel (both must publish).

Keyboard:  c = start an episode (record + teleop),  s = stop episode,  q = exit.
"""
import os
import time
import argparse
import datetime
import sys
import subprocess
from threading import Event

from paradex.dataset_acqusition.capture import CaptureSession
from paradex.dataset_acqusition.match_sync import postprocess_session
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.path import shared_dir, local_shared_dir

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--arm', type=str, default="xarm")
parser.add_argument('--hand', type=str, default="inspire")
parser.add_argument('--hand-side', type=str, default="right", choices=["right", "bimanual"])
parser.add_argument('--capture-root', type=str, default="capture/hri_vive")
parser.add_argument('--no-camera', dest='camera', action='store_false',
                    help='Disable remote cameras (teleop-only).')
parser.set_defaults(camera=True)
parser.add_argument('--no-clutch', '--no_clutch', dest='clutch', action='store_false',
                    help='Disable the left-hand clutch; robot always follows the right hand.')
parser.set_defaults(clutch=True)
parser.add_argument('--no-limit', '--no_limit', dest='limit_wrist', action='store_false',
                    help='Disable the wrist velocity/glitch-rejection limiter '
                         '(by default ON: rejects VIVE tracking spikes so the arm '
                         'does not lurch).')
parser.set_defaults(limit_wrist=True)
parser.add_argument('--calib', action='store_true',
                    help='Print VIVE->robot translation deltas for frame-alignment calibration.')
parser.add_argument('--collect', action='store_true',
                    help='On quit, collect capture-PC videos (transcode+pull) but SKIP '
                         'postprocess (do that later with finalize.py).')
parser.add_argument('--finalize', action='store_true',
                    help='On quit, collect videos AND postprocess arm/hand onto the camera '
                         'timeline. SLOW; OFF by default (run finalize.py later).')
parser.add_argument('--video-dest', '--video_dest', dest='video_dest',
                    choices=['local', 'nas'], default='local',
                    help='Where collect puts the videos: "local" pulls them to this main '
                         'PC data2 store next to the robot data (no NAS, default); "nas" '
                         'pushes everything to the shared NAS (slow).')
args = parser.parse_args()

stop_event = Event()
save_event = Event()
exit_event = Event()
listen_keyboard({"c": save_event, "q": exit_event, "s": stop_event})
print("Keyboard control: c=start episode, s=stop, q=exit")

# Remote cameras run free-run (no UTGE900). The main PC here has no local
# timestamp camera, so the timestamp monitor is disabled as well.
cs = CaptureSession(
    camera=args.camera,
    arm=args.arm,
    hand=args.hand,
    teleop="vive",
    hand_ip=True,               # Inspire over Modbus TCP (network.json inspire_ip)
    hand_side=args.hand_side,
    use_sync_gen=False,
    use_timestamp_monitor=False,
    limit_wrist=args.limit_wrist,
)

# With the clutch off, fully ignore the LEFT glove from the very start: its
# frames are dropped in the receiver (no NaN-spam warnings) and get_data no
# longer waits on it, so a dead/NaN left glove can't stall right-hand teleop.
# (This runner keeps gesture-exit off, so the left glove is only the clutch.)
if getattr(cs, "teleop_device", None) is not None and hasattr(cs.teleop_device, "set_require_left_hand"):
    cs.teleop_device.set_require_left_hand(args.clutch)
    print(f"[capture] left glove {'required (clutch on)' if args.clutch else 'IGNORED (clutch off)'}")

if args.calib and getattr(cs, "retargetor", None) is not None:
    cs.retargetor._debug_translation = True
    print("[calib] Move the RIGHT hand along one axis at a time and read "
          "'VIVE dt' -> map hand-forward/left/up to robot +X/+Y/+Z.")

recorded = []          # session leaf names captured this run
# avoid doubling when --name already includes the capture-root prefix
_croot = args.capture_root.rstrip("/")
if args.name == _croot or args.name.startswith(_croot + "/"):
    save_root = args.name
else:
    save_root = os.path.join(args.capture_root, args.name)

try:
    while not exit_event.is_set():
        if not save_event.is_set():
            stop_event.clear()
            time.sleep(0.01)
            continue

        save_event.clear()
        stop_event.clear()

        index = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        rel_path = os.path.join(save_root, index)
        cs.start(rel_path)
        print("Recording + teleop:", rel_path, "(s=stop, q=exit)")

        state = cs.teleop(stop_event=stop_event, exit_event=exit_event,
                          use_gesture_exit=False, clutch=args.clutch)
        cs.stop()
        print("Stopped episode:", rel_path)
        recorded.append(index)

        if state == "exit":
            break
finally:
    cs.end()
    print("Exiting teleoperation recording.")

# --collect: pull videos on quit.  --finalize: also postprocess.
# Default video-dest=local -> everything stays on this main PC's data2 store next
# to the robot data (no NAS). Default action: neither (fast quit) -> collect later.
data_root = local_shared_dir if args.video_dest == "local" else shared_dir
if (args.collect or args.finalize) and args.camera and recorded:
    print(f"[VIDEO] Collecting videos for {save_root} (dest={args.video_dest})...")
    collect_script = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                  "src", "process", "teleop_real", "collect_videos.py")
    subprocess.run([sys.executable, os.path.normpath(collect_script),
                    "--save_path", save_root, "--dest", args.video_dest,
                    "--sessions", *recorded], check=False)
    print(f"[VIDEO] Done. Videos in {data_root}/{save_root}/<session>/videos/")
    if args.finalize:
        for idx in recorded:
            print(f"[POST] postprocess (camera timeline): {idx}")
            postprocess_session(os.path.join(data_root, save_root, idx))
    else:
        print(f"[NEXT] postprocess later: "
              f"python src/process/teleop_real/finalize.py --save_path {save_root}")
elif recorded:
    print(f"[NEXT] Collect videos + postprocess later:")
    print(f"       python src/process/teleop_real/finalize.py --save_path {save_root}")
