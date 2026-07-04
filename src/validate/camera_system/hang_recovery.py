"""Hardware validation for the P4 camera-hang fix.

Background (design/camera-recording-redesign.md, P4): when frames stop arriving
(LAN drop, no hardware trigger, etc.) the camera used to block forever in
GetNextImage(), so Camera.stop()/end() — and therefore the whole daemon — hung
and cameras could not be restarted without a hard kill.

The fix makes get_image() time out (GRAB_TIMEOUT_MS) and stop()/end() use finite
waits. This script proves it on real hardware: it induces "no frames" and checks
that stop()/end() return within a bounded time instead of hanging.

Each timed call runs under a watchdog thread, so if the OLD (broken) code is
running the script reports "HANG DETECTED (FAIL)" and exits instead of hanging.

Run on a CAPTURE PC (needs PySpin + cameras):

    # Test 1 needs NO UTGE900 trigger running (that is how we simulate frame loss).
    python src/validate/camera_system/hang_recovery.py
    python src/validate/camera_system/hang_recovery.py --interactive   # + LAN-pull test

After a run, if a camera was left mid-acquisition, recover with:
    python src/camera/reset_cameras.py            # (from the main PC)
"""
import argparse
import threading
import time

from paradex.io.camera_system.camera_loader import CameraLoader
from paradex.io.camera_system.pyspin import GRAB_TIMEOUT_MS

GRAB_TIMEOUT_S = GRAB_TIMEOUT_MS / 1000.0
# stop()/end() should return within a few grab timeouts even across N cameras.
BOUND_S = GRAB_TIMEOUT_S * 3 + 5.0


def run_bounded(label, fn):
    """Run fn() under a watchdog. Returns (passed, elapsed_or_None).

    If fn does not return within BOUND_S it is a hang (the bug) — we report FAIL
    and leave the thread as a daemon so this script can still exit."""
    result = {}
    def _run():
        t0 = time.time()
        try:
            fn()
        finally:
            result["dt"] = time.time() - t0
    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(timeout=BOUND_S + 2.0)
    if th.is_alive():
        print(f"    {label}: STILL RUNNING after {BOUND_S + 2.0:.0f}s -> HANG DETECTED (FAIL)")
        return False, None
    dt = result.get("dt")
    ok = dt is not None and dt < BOUND_S
    print(f"    {label}: returned in {dt:.2f}s -> {'PASS' if ok else 'FAIL'} (bound {BOUND_S:.0f}s)")
    return ok, dt


def test_no_trigger(camera):
    """sync mode with no trigger => no frames arrive => stop() must not hang.
    This is the core P4 regression check and needs no cable-pulling."""
    print("[Test 1] sync mode, NO trigger running (simulated frame loss)")
    print("  -> make sure the UTGE900 trigger is OFF, so no frames arrive.")
    camera.start("stream", True)          # syncMode=True, no save
    print("  acquiring 3s (expect ~0 frames)...")
    time.sleep(3.0)
    return run_bounded("stop()", camera.stop)[0]


def test_normal(camera):
    """Baseline: free-run (frames flowing), stop() should be quick. No trigger needed."""
    print("[Test 2] free-run (frames flowing), normal stop")
    camera.start("stream", False, fps=30)
    time.sleep(2.0)
    return run_bounded("stop()", camera.stop)[0]


def test_lan_pull(camera):
    """Interactive: pull a camera's LAN cable mid-capture, then stop must not hang."""
    print("[Test 3] interactive LAN-drop")
    camera.start("stream", False, fps=30)
    input("  Frames are streaming. PULL a camera's LAN cable, then press Enter to stop... ")
    return run_bounded("stop()", camera.stop)[0]


def main():
    parser = argparse.ArgumentParser(description="Validate the P4 camera-hang fix on hardware.")
    parser.add_argument("--interactive", action="store_true",
                        help="also run the interactive LAN-pull test")
    args = parser.parse_args()

    print(f"GRAB_TIMEOUT_MS={GRAB_TIMEOUT_MS}  hang bound={BOUND_S:.0f}s\n")
    camera = CameraLoader()
    results = {}

    results["normal_stop"] = test_normal(camera)
    results["no_trigger_stop"] = test_no_trigger(camera)
    if args.interactive:
        results["lan_pull"] = test_lan_pull(camera)
    # end() must also return (not hang) even after the above.
    results["end_no_hang"] = run_bounded("end()", camera.end)[0]

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    all_ok = all(results.values())
    print("\nALL PASS — P4 hang fix verified." if all_ok
          else "\nSOME FAILED — cameras may hang on frame loss (check the fix / run reset_cameras.py).")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
