"""Main-PC recovery: hard-reset the camera daemons on every capture PC.

Use this when cameras hang (LAN drop / frame loss) and a plain `pkill run_auto`
on the main PC leaves the capture-PC `server_daemon.py` stuck — so the next run
can't start cameras (the wedged CameraLoader blocks register/start).

This force-kills (`-9`) the daemon on each capture PC and relaunches it clean.
`-9` is used because a thread wedged in a native Spinnaker call won't respond to
SIGTERM. See design/camera-recording-redesign.md P4.

Usage:
    python src/camera/reset_cameras.py                      # all capture PCs
    python src/camera/reset_cameras.py --pc_list capture1 capture2
    python src/camera/reset_cameras.py --no_restart         # only kill, don't relaunch
"""
import argparse
import subprocess
import time

from paradex.io.capture_pc.ssh import run_script, ssh_port
from paradex.utils.system import get_pc_list, get_pc_ip

DAEMON = "src/camera/server_daemon.py"
MONITOR = "src/camera/monitor_daemon.py"


def kill_daemons(pc_list):
    """SIGKILL the camera daemons on each capture PC (SIGTERM can't interrupt a
    thread wedged in a native GetNextImage)."""
    for pc_name in pc_list:
        ip = get_pc_ip(pc_name)
        remote_cmd = f"pkill -9 -f '{DAEMON}'; pkill -9 -f '{MONITOR}'; sleep 0.5"
        ssh_cmd = f"ssh -p {ssh_port} {pc_name}@{ip} \"{remote_cmd}\""
        r = subprocess.run(ssh_cmd, shell=True)
        # pkill returns 1 when nothing matched — not an error for us.
        print(f"[{pc_name}] killed camera daemons (rc={r.returncode})")


def main():
    parser = argparse.ArgumentParser(description="Hard-reset camera daemons on capture PCs.")
    parser.add_argument("--pc_list", type=str, nargs="+", default=None,
                        help="Capture PCs to reset (default: all from system config).")
    parser.add_argument("--no_restart", action="store_true",
                        help="Only kill the daemons; do not relaunch.")
    args = parser.parse_args()

    pc_list = args.pc_list if args.pc_list else get_pc_list()
    print(f"[reset] capture PCs: {pc_list}")

    kill_daemons(pc_list)

    if args.no_restart:
        print("[reset] --no_restart set; daemons killed, not relaunched.")
        return

    time.sleep(1.0)
    print("[reset] relaunching server_daemon.py ...")
    run_script("python src/camera/server_daemon.py", pc_list)
    print("[reset] done. Verify with monitor_daemon or remote_camera_controller.")


if __name__ == "__main__":
    main()
