"""Smoke-test the Inspire LEFT hand over Modbus TCP (IP).

Single-hand operation: left and right share the same IP/port (only one hand
is connected at a time), so this uses the existing `inspire` network config.

Sequence: HOME -> OPEN -> CLOSED -> OPEN -> per-finger sinusoidal wave -> HOME.
Prints commanded vs. reported angles after each pose.

Usage:
    python src/validate/robot/inspire_left.py
"""
import time

import numpy as np

from paradex.utils.system import network_info
from paradex.io.robot_controller.inspire_controller_ip import InspireControllerIP


# Inspire angle commands: 0 = fully closed, 1000 = fully open
HOME   = np.array([800, 800, 800, 800, 800, 800], dtype=np.int32)
OPEN   = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.int32)
CLOSED = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
DOF_LABELS = ("little", "ring", "middle", "index", "thumb_bend", "thumb_rot")
WAVE_SECS = 8.0


def _print_state(label, target, ic):
    qpos = ic.get_qpos()
    pairs = "  ".join(f"{n}={int(v):>4d}" for n, v in zip(DOF_LABELS, qpos))
    print(f"[{label}] target={target.tolist()}\n          actual: {pairs}")


def main():
    print(f"Connecting to Inspire LEFT at "
          f"{network_info['inspire']['ip']}:{network_info['inspire']['port']} ...")
    ic = InspireControllerIP(**network_info["inspire"])
    print("Connected.")

    try:
        for label, target, hold in [("HOME", HOME, 2.0),
                                    ("OPEN", OPEN, 1.5),
                                    ("CLOSED", CLOSED, 1.5),
                                    ("OPEN", OPEN, 1.5)]:
            ic.move(target)
            time.sleep(hold)
            _print_state(label, target, ic)

        print(f"--> WAVE (per-finger sinusoid, {WAVE_SECS}s)")
        t0 = time.time()
        freq = 0.3  # Hz
        while time.time() - t0 < WAVE_SECS:
            t = time.time() - t0
            cmd = np.array(
                [int(500 + 400 * np.sin(2 * np.pi * freq * t - i * (2 * np.pi / 6)))
                 for i in range(6)],
                dtype=np.int32,
            )
            ic.move(cmd)
            time.sleep(0.05)
        _print_state("WAVE END", cmd, ic)

        ic.move(HOME)
        time.sleep(2.0)
        _print_state("HOME", HOME, ic)
    finally:
        ic.end()


if __name__ == "__main__":
    main()
