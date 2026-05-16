"""
Diagnose how human-wrist motion maps to robot translation axes.

Why: Retargetor adds the right-wrist position delta (pelvis frame, from
XSensReceiver) DIRECTLY to the robot base frame -- device2global is loaded but
never applied. So if you stand at an angle to the robot, some directions get
inverted/swapped. This tool measures the actual mapping empirically.

Usage:
    python src/validate/teleop/wrist_axis_check.py
    - press 'z' to zero (capture reference pose)
    - then move your RIGHT hand purely in ONE direction ~20-30cm and hold
    - read which axis changed and its sign
    - repeat for: forward, left, up
    - 'q' to quit

Report the 3 lines (forward / left / up) and we bake the correction into
DEVICE2GLOBAL + apply it in unimanual.py.
"""
import time
import numpy as np
from threading import Event

from paradex.io.teleop import XSensReceiver, xsens_joint_name  # noqa: F401
from paradex.utils.system import network_info
from paradex.utils.keyboard_listener import listen_keyboard


def main():
    recv = XSensReceiver(**network_info["xsens"]["param"])
    zero_evt = Event()
    quit_evt = Event()
    listen_keyboard({"z": zero_evt, "q": quit_evt})
    print("[wrist_axis_check] move RIGHT hand. 'z'=zero, 'q'=quit")

    ref = None
    while not quit_evt.is_set():
        data = recv.get_data()
        if data["Right"] is None:
            time.sleep(0.05)
            continue
        p = data["Right"]["wrist"][:3, 3].copy()

        if zero_evt.is_set():
            ref = p.copy()
            zero_evt.clear()
            print(f"\n[ZERO] ref = [{ref[0]:+.3f} {ref[1]:+.3f} {ref[2]:+.3f}] "
                  f"(pelvis frame, meters)")

        if ref is not None:
            d = p - ref
            ax = int(np.argmax(np.abs(d)))
            names = ["X", "Y", "Z"]
            sign = "+" if d[ax] >= 0 else "-"
            print(f"\rdelta x={d[0]:+.3f} y={d[1]:+.3f} z={d[2]:+.3f}  "
                  f"| dominant: {sign}{names[ax]} ({d[ax]:+.3f} m)      ",
                  end="", flush=True)
        time.sleep(0.05)

    recv.quit()
    print("\n[wrist_axis_check] done")


if __name__ == "__main__":
    main()
