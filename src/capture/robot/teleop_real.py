import datetime
import os
import sys
import subprocess
import argparse
from threading import Event

from paradex.dataset_acqusition.capture import CaptureSession
from paradex.dataset_acqusition.match_sync import postprocess_session
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.path import shared_dir


parser = argparse.ArgumentParser()

parser.add_argument('--device', choices=['xsens', 'occulus'])
parser.add_argument('--arm', type=str, default=None)
parser.add_argument('--hand', type=str, default=None)
parser.add_argument('--hand_usb', action='store_true',
                    help="Use USB (Modbus-RTU) mode for Inspire hand. Default: IP (Modbus-TCP).")
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--camera', action='store_true',
                    help="Capture Camera Video as well")
parser.add_argument('--use_sync_gen', action='store_true',
                    help="Use UTGE900 hardware sync generator. Default: free-run (no sync gen).")
parser.add_argument('--camera_fps', type=int, default=30,
                    help="Camera FPS (and sync gen freq if --use_sync_gen). Default 30.")
parser.add_argument('--no_timestamp_monitor', action='store_true',
                    help="Skip the main-PC TimestampMonitor (no local PySpin camera needed). "
                         "Lowers camera<->robot time-sync precision.")
parser.add_argument('--no_collect', action='store_true',
                    help="Skip auto rsync of capture-PC videos + re-postprocess on quit. "
                         "Run src/process/teleop_real/collect_videos.py manually later.")
parser.add_argument('--translation_scale', type=float, default=1.5,
                    help="Amplify hand->robot translation (robot_dt = scale*human_dt). "
                         "Default 1.5: small hand motion -> larger robot motion.")
parser.add_argument('--no_gesture', action='store_true',
                    help="Disable V/spider gesture start-stop/exit. Keyboard s/e/q only. "
                         "(any non-open left hand still acts as clutch)")

args = parser.parse_args()

start_event = Event()
stop_event = Event()
exit_event = Event()

listen_keyboard({
    "s": start_event,
    "e": stop_event,
    "q": exit_event,
})
print("Keys: [s]=start recording  [e]=end recording  [q]=quit")
if args.no_gesture:
    print("Left hand: open=follow  any other=clutch(pause)  [gestures OFF]")
else:
    print("Left hand: open=follow  fist=clutch(pause)  V(~1s)=start/stop  spider(~1s)=quit")

cs = CaptureSession(
    camera=args.camera,
    arm=args.arm,
    hand=args.hand,
    teleop=args.device,
    hand_ip=not args.hand_usb,
    use_sync_gen=args.use_sync_gen,
    camera_fps=args.camera_fps,
    use_timestamp_monitor=not args.no_timestamp_monitor,
    translation_scale=args.translation_scale,
)

recorded_sessions = []

while not exit_event.is_set():
    print("Waiting for [s] to start a new recording session...")
    state = cs.teleop(
        stop_event=start_event,
        exit_event=exit_event,
        use_gesture_exit=not args.no_gesture,
    )
    if state == "exit":
        break

    name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cs.start(os.path.join(args.save_path, name))
    print(f"[REC] Started: {name}")

    state = cs.teleop(
        stop_event=stop_event,
        exit_event=exit_event,
        use_gesture_exit=not args.no_gesture,
    )

    cs.stop()
    print(f"[REC] Stopped: {name}")

    session_dir = os.path.join(shared_dir, args.save_path, name)
    recorded_sessions.append(session_dir)
    print(f"[POST] Postprocessing (sensor-only, videos not yet collected): {session_dir}")
    postprocess_session(session_dir)
    print(f"[POST] Done: {session_dir}")

print("Exiting teleoperation recording.")
cs.end()

if args.camera and recorded_sessions and args.no_collect:
    print(f"[VIDEO] --no_collect: skipping auto rsync. Run later:")
    print(f"        python src/process/teleop_real/collect_videos.py --save_path {args.save_path}")
    print(f"        then re-run postprocess for synthesized camera timeline.")

if args.camera and recorded_sessions and not args.no_collect:
    print(f"[VIDEO] Collecting videos for {args.save_path} (rsync + mp4)...")
    collect_script = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "process", "teleop_real", "collect_videos.py",
    )
    session_names = [os.path.basename(p) for p in recorded_sessions]
    subprocess.run(
        [sys.executable, os.path.normpath(collect_script),
         "--save_path", args.save_path,
         "--sessions", *session_names],
        check=False,
    )
    print(f"[VIDEO] Done. Videos in {shared_dir}/{args.save_path}/<session>/videos/")

    # Videos now exist -> re-postprocess so synthesize_camera_timeline() kicks
    # in and arm/hand get resampled onto the video-frame timeline (frame i <-> action i).
    for session_dir in recorded_sessions:
        print(f"[POST] Re-postprocessing with synthesized camera timeline: {session_dir}")
        postprocess_session(session_dir)

    print(f"[VIDEO] Raw .avi kept. To reclaim space: "
          f"python src/process/teleop_real/delete_raw_avi.py --save_path {args.save_path} --yes")