#!/usr/bin/env python3
"""
Demo script for the Inspire Hand via IP connection.
Demonstrates connection, pose reading/writing, force reading, and tactile sensor reading.
"""

import sys
import time
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from paradex.utils.system import network_info

# Add parent directory to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inspire_controller_ip import InspireControllerIP


def parse_args():
    parser = argparse.ArgumentParser(description="Inspire Hand IP Demo")
    parser.add_argument(
        "--tactile", "-t",
        action="store_true",
        help="Enable tactile sensor reading"
    )
    parser.add_argument(
        "--demo", "-d",
        type=str,
        choices=["basic", "pose", "force", "full", "tactile", "finger", "gui"],
        default="basic",
        help="Demo to run (default: basic)"
    )
    parser.add_argument(
        "--finger", type=str, default=None,
        help="(finger demo) finger name/index for one-shot. "
             "names: little/ring/middle/index/thumb_bend/thumb_rot "
             "(aliases: l/r/m/i/tb/tr, pinky=little, thumb=thumb_rot, 0-5 ok)"
    )
    parser.add_argument(
        "--angle", type=int, default=None,
        help="(finger demo) angle 0(close)~2000(open) for one-shot"
    )
    parser.add_argument(
        "--base", type=int, default=None,
        help="(finger demo) baseline angle for unmoved fingers when starting "
             "from a fresh pose (skip to keep current pose)"
    )
    return parser.parse_args()


def print_pose(hand: InspireControllerIP):
    """Print the current pose of the hand."""
    try:
        angles = hand.get_qpos()

        print("\n=== Current Pose ===")
        print(f"Angle values: {angles}")

        finger_names = ["Little", "Ring", "Middle", "Index", "Thumb Bend", "Thumb Rotate"]
        for i, name in enumerate(finger_names):
            print(f"  {name}: angle={angles[i]}")
    except Exception as e:
        print(f"Error reading pose: {e}")


def print_force_info(hand: InspireControllerIP):
    """Print force information from the hand."""
    try:
        forces = hand.get_force()

        print("\n=== Force Information ===")
        print(f"Force values: {forces}")

        finger_names = ["Little", "Ring", "Middle", "Index", "Thumb Bend", "Thumb Rotate"]
        for i, name in enumerate(finger_names):
            print(f"  {name}: force={forces[i]}")
    except Exception as e:
        print(f"Error reading force info: {e}")


def basic_demo(hand: InspireControllerIP):
    """Run a basic demo showing connection, pose reading, and pose writing."""
    print("\n=== Basic Demo ===")

    # Read current pose
    print("\n1. Reading current pose...")
    print_pose(hand)

    # Open all fingers (higher values = open)
    print("\n2. Opening all fingers (angles = [2000, 2000, 2000, 2000, 2000, 2000])...")
    hand.move(np.array([2000, 2000, 2000, 2000, 2000, 2000]))
    time.sleep(2)

    # Read pose again
    print("\n3. Reading pose after opening...")
    print_pose(hand)

    # Close all fingers (lower values = closed)
    print("\n4. Closing all fingers (angles = [0, 0, 0, 0, 0, 0])...")
    hand.move(np.array([0, 0, 0, 0, 0, 0]))
    time.sleep(2)

    # Read pose again
    print("\n5. Reading pose after closing...")
    print_pose(hand)

    # Set to middle position
    print("\n6. Setting to middle position (angles = [500, 500, 500, 500, 500, 500])...")
    hand.move(np.array([500, 500, 500, 500, 500, 500]))
    time.sleep(2)

    # Read final pose
    print("\n7. Final pose:")
    print_pose(hand)

    # Return to home
    print("\n8. Returning to home position...")
    hand.home_robot()
    time.sleep(2)


def pose_demo(hand: InspireControllerIP):
    """Demonstrate pose reading and writing."""
    print("\n=== Pose Demo ===")

    # Read initial pose
    print("\nInitial pose:")
    print_pose(hand)

    # Try different poses
    poses = [
        (np.array([2000, 2000, 2000, 2000, 2000, 2000]), "Fully open"),
        (np.array([0, 0, 0, 0, 0, 0]), "Fully closed"),
        (np.array([1000, 1000, 1000, 1000, 1000, 1000]), "Middle position"),
        (np.array([500, 1500, 500, 1500, 500, 1000]), "Alternating pattern"),
    ]

    for pose, description in poses:
        print(f"\nSetting pose: {description}")
        print(f"  Values: {pose}")
        hand.move(pose)
        time.sleep(2)

        print("Current pose after setting:")
        print_pose(hand)
        time.sleep(1)

    # Return to home
    print("\nReturning to home position...")
    hand.home_robot()
    time.sleep(2)


def force_demo(hand: InspireControllerIP):
    """Demonstrate force information reading."""
    print("\n=== Force Information Demo ===")

    print("\n1. Reading force information (hand should be open, no force)...")
    hand.move(np.array([2000, 2000, 2000, 2000, 2000, 2000]))
    time.sleep(1)
    print_force_info(hand)

    print("\n2. Closing fingers to create force...")
    hand.move(np.array([0, 0, 0, 0, 0, 0]))
    time.sleep(2)

    print("\n3. Reading force information after closing...")
    print_force_info(hand)

    print("\n4. Opening fingers...")
    hand.move(np.array([2000, 2000, 2000, 2000, 2000, 2000]))
    time.sleep(2)

    print("\n5. Reading force information after opening...")
    print_force_info(hand)

    # Monitor force in real-time
    print("\n6. Monitoring force information for 10 readings (press Ctrl+C to stop)...")
    try:
        for i in range(10):
            print(f"\n--- Reading {i+1}/10 ---")
            print_force_info(hand)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

    # Return to home
    print("\nReturning to home position...")
    hand.home_robot()
    time.sleep(2)


def full_demo(hand: InspireControllerIP):
    """Run a comprehensive demo showing all features."""
    print("\n=== Full Demo ===")

    # Set speed and force
    print("\n1. Setting speed to 1000...")
    hand.write6('speedSet', [1000, 1000, 1000, 1000, 1000, 1000])

    print("\n2. Setting force threshold to 400...")
    hand.write6('forceSet', [400, 400, 400, 400, 400, 400])

    # Read initial status
    print("\n3. Reading initial pose...")
    print_pose(hand)

    # Force operations
    print("\n4. Force operations...")
    print_force_info(hand)

    print("\n5. Moving through a sequence...")
    sequence = [
        (np.array([2000, 2000, 2000, 2000, 2000, 2000]), "Fully Open"),
        (np.array([1000, 1000, 1000, 1000, 1000, 1000]), "Middle Position"),
        (np.array([0, 0, 0, 0, 0, 0]), "Fully Closed"),
        (np.array([500, 1500, 500, 1500, 500, 1000]), "Alternating Pattern"),
    ]

    for pose, name in sequence:
        print(f"\n   Moving to: {name}")
        hand.move(pose)
        time.sleep(2)

        print("   Current pose:")
        angles = hand.get_qpos()
        print(f"     {angles}")

        print("   Force info:")
        forces = hand.get_force()
        print(f"     {forces}")

    # Test individual finger movements
    print("\n6. Testing individual finger movements...")
    for idx in range(6):
        finger_names = ["Little", "Ring", "Middle", "Index", "Thumb Bend", "Thumb Rotate"]
        print(f"\n   Moving {finger_names[idx]}...")

        # Start with all open
        base_pose = np.array([2000, 2000, 2000, 2000, 2000, 2000])
        hand.move(base_pose)
        time.sleep(1)

        # Move this finger to closed
        test_pose = base_pose.copy()
        test_pose[idx] = 0
        hand.move(test_pose)
        time.sleep(1.5)

        print(f"   Current pose:")
        angles = hand.get_qpos()
        print(f"     {angles}")

    # Return to home
    print("\n7. Returning to home position...")
    hand.home_robot()
    time.sleep(2)


FINGER_NAMES = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]
FINGER_ALIAS = {
    "little": 0, "l": 0, "pinky": 0, "0": 0,
    "ring":   1, "r": 1, "1": 1,
    "middle": 2, "m": 2, "2": 2,
    "index":  3, "i": 3, "3": 3,
    "thumb_bend": 4, "tb": 4, "thumb_pitch": 4, "thumb1": 4, "4": 4,
    "thumb_rot":  5, "tr": 5, "thumb_yaw":   5, "thumb":  5, "thumb2": 5, "5": 5,
}


def _resolve_finger(s):
    if s is None:
        return None
    return FINGER_ALIAS.get(str(s).strip().lower())


def _move_one(hand, idx, angle, base=None):
    """Set finger `idx` to `angle`, keep others. If base given, reset others to base first."""
    angle = int(np.clip(angle, 0, 2000))
    if base is not None:
        cur = np.full(6, int(np.clip(base, 0, 2000)), dtype=np.int32)
    else:
        cur = np.asarray(hand.get_qpos(), dtype=np.int32).copy()
    cur[idx] = angle
    hand.move(cur)
    return cur


def finger_demo(hand: InspireControllerIP, init_finger=None, init_angle=None, base=None):
    """Move one finger at a time. One-shot (CLI args) or interactive loop."""
    print("\n=== Finger Demo ===")
    print(f"fingers (idx: name): " +
          ", ".join(f"{i}:{n}" for i, n in enumerate(FINGER_NAMES)))
    print("angle 0=closed, 2000=open")

    if init_finger is not None and init_angle is not None:
        idx = _resolve_finger(init_finger)
        if idx is None:
            print(f"unknown finger: {init_finger}")
            return
        _move_one(hand, idx, init_angle, base=base)
        time.sleep(1.5)
        print_pose(hand)
        return

    print("\nCommands:")
    print("  <finger> <angle>   e.g. 'index 500' or 'i 1500' or '3 0'")
    print("  open | close | home | pose")
    print("  base <val>         set baseline for fresh pose (others=val on next move)")
    print("  q | quit           exit")

    cur_base = base
    while True:
        try:
            s = input(">>> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not s:
            continue
        if s in ("q", "quit", "exit"):
            break
        if s == "home":
            hand.home_robot()
            time.sleep(1)
            print_pose(hand)
            continue
        if s == "open":
            hand.move(np.full(6, 2000, dtype=np.int32))
            time.sleep(1)
            print_pose(hand)
            continue
        if s == "close":
            hand.move(np.zeros(6, dtype=np.int32))
            time.sleep(1)
            print_pose(hand)
            continue
        if s == "pose":
            print_pose(hand)
            continue
        parts = s.split()
        if len(parts) == 2 and parts[0] == "base":
            try:
                cur_base = int(parts[1])
                print(f"[finger] baseline -> {cur_base} (applied on next move)")
            except ValueError:
                print("base needs int")
            continue
        if len(parts) != 2:
            print("usage: <finger> <angle>  |  open | close | home | pose | base N | q")
            continue
        idx = _resolve_finger(parts[0])
        if idx is None:
            print(f"unknown finger: {parts[0]}  (try: {FINGER_NAMES})")
            continue
        try:
            a = int(parts[1])
        except ValueError:
            print("angle must be int 0-2000")
            continue
        sent = _move_one(hand, idx, a, base=cur_base)
        cur_base = None  # apply base only once
        time.sleep(0.8)
        print(f"[finger] sent {FINGER_NAMES[idx]}={a}  full={sent.tolist()}")
        print_pose(hand)


def gui_demo(hand: InspireControllerIP):
    """Tkinter slider GUI: 6 fingers, 0(closed) ~ 2000(open)."""
    import tkinter as tk

    root = tk.Tk()
    root.title("Inspire Hand")

    status_var = tk.StringVar(value="ready")
    actual_var = tk.StringVar(value="actual: -")
    live_var = None  # set later

    sliders = []
    cmd_vars = []

    def send_move(_evt=None):
        cur = np.array([v.get() for v in cmd_vars], dtype=np.int32)
        try:
            hand.move(cur)
            status_var.set(f"cmd: {cur.tolist()}")
        except Exception as e:
            status_var.set(f"move err: {e}")

    def read_pose(sync_sliders=True):
        try:
            cur = np.asarray(hand.get_qpos(), dtype=np.int32)
            actual_var.set(f"actual: {cur.tolist()}")
            if sync_sliders:
                for v, c in zip(cmd_vars, cur):
                    v.set(int(c))
        except Exception as e:
            actual_var.set(f"read err: {e}")

    def set_all(val):
        for v in cmd_vars:
            v.set(int(val))
        send_move()

    def go_home():
        hand.home_robot()
        root.after(800, lambda: read_pose(sync_sliders=True))

    # Sliders (one per finger)
    for i, name in enumerate(FINGER_NAMES):
        row = tk.Frame(root)
        row.pack(fill="x", padx=8, pady=3)
        tk.Label(row, text=f"{i}: {name}", width=16, anchor="w").pack(side="left")
        v = tk.IntVar(value=1000)
        s = tk.Scale(row, from_=0, to=2000, orient="horizontal",
                     resolution=10, length=420, variable=v)
        s.pack(side="left", fill="x", expand=True)
        s.bind("<ButtonRelease-1>", send_move)
        sliders.append(s)
        cmd_vars.append(v)

    # Buttons
    btn = tk.Frame(root)
    btn.pack(fill="x", padx=8, pady=6)
    tk.Button(btn, text="Open (2000)", command=lambda: set_all(2000)).pack(side="left", padx=3)
    tk.Button(btn, text="Mid  (1000)", command=lambda: set_all(1000)).pack(side="left", padx=3)
    tk.Button(btn, text="Close ( 0 )", command=lambda: set_all(0)).pack(side="left", padx=3)
    tk.Button(btn, text="Home", command=go_home).pack(side="left", padx=3)
    tk.Button(btn, text="Read Pose", command=lambda: read_pose(sync_sliders=True)).pack(side="left", padx=3)
    tk.Button(btn, text="Quit", command=root.destroy).pack(side="right", padx=3)

    # Live readout (polls actual angles ~5 Hz)
    live_var = tk.BooleanVar(value=True)
    tk.Checkbutton(btn, text="Live actual", variable=live_var).pack(side="right", padx=8)

    def poll():
        if live_var.get():
            read_pose(sync_sliders=False)
        root.after(200, poll)

    tk.Label(root, textvariable=status_var, anchor="w").pack(fill="x", padx=8)
    tk.Label(root, textvariable=actual_var, anchor="w").pack(fill="x", padx=8, pady=(0, 6))
    tk.Label(root, text="drag a slider then release to send. Live shows measured pose.",
             anchor="w", fg="#555").pack(fill="x", padx=8, pady=(0, 6))

    read_pose(sync_sliders=True)  # init from current
    poll()
    root.mainloop()


def tactile_demo(hand: InspireControllerIP):
    """Demonstrate tactile information reading."""
    if not hand.tactile:
        print("\nTactile mode is not enabled. Please run with --tactile flag.")
        return

    print("\n=== Tactile Information Demo ===")
    print("\n1. Opening hand...")
    hand.move(np.array([2000, 2000, 2000, 2000, 2000, 2000]))
    time.sleep(1)

    print("\n2. Reading tactile data...")
    try:
        tactile_data = hand.get_tactile()

        print("\nTactile sensor data:")
        for name, data in tactile_data.items():
            print(f"\n{name}:")
            print(f"  Shape: {data.shape}")
            print(f"  Min: {data.min()}, Max: {data.max()}, Mean: {data.mean():.2f}")
            print(f"  Data:\n{data}")
    except Exception as e:
        print(f"Error reading tactile data: {e}")
        import traceback
        traceback.print_exc()

    # Optional: Visualize tactile data
    print("\n3. Would you like to visualize tactile data in real-time? (yes/no)")
    # For demo purposes, we'll skip interactive input
    # In a real scenario, you could add visualization similar to hand_demo.py

    print("\n4. Closing fingers to test contact...")
    hand.move(np.array([200, 200, 200, 200, 200, 200]))
    time.sleep(2)

    print("\n5. Reading tactile data after closing...")
    try:
        tactile_data = hand.get_tactile()

        print("\nTactile sensor data after closing:")
        for name, data in tactile_data.items():
            print(f"\n{name}:")
            print(f"  Min: {data.min()}, Max: {data.max()}, Mean: {data.mean():.2f}")
    except Exception as e:
        print(f"Error reading tactile data: {e}")

    # Return to home
    print("\nReturning to home position...")
    hand.home_robot()
    time.sleep(2)


def main():
    """Main demo function."""
    args = parse_args()

    try:
        # network_config = json.load(open(os.path.join(config_dir, "network.json"), "r"))
        # self.ip = network_config["inspire_ip"]["param"]["ip"]
        # self.port = network_config["inspire_ip"]["param"]["port"]
        
        ip, port = network_info['inspire']['ip'], network_info['inspire']['port']
        # Create hand instance
        print(f"Connecting to Inspire Hand at {ip}:{port}...")
        hand = InspireControllerIP(
            ip=ip, port=port,
            tactile=args.tactile
        )

        print("Connected successfully!")
        print(f"Tactile mode: {'Enabled' if args.tactile else 'Disabled'}")

        # Wait a moment for the hand to initialize
        time.sleep(1)

        # Run the selected demo
        if args.demo == "basic":
            basic_demo(hand)
        elif args.demo == "pose":
            pose_demo(hand)
        elif args.demo == "force":
            force_demo(hand)
        elif args.demo == "full":
            full_demo(hand)
        elif args.demo == "tactile":
            tactile_demo(hand)
        elif args.demo == "finger":
            finger_demo(hand,
                        init_finger=args.finger,
                        init_angle=args.angle,
                        base=args.base)
        elif args.demo == "gui":
            gui_demo(hand)

        print("\n=== Demo completed successfully! ===")

        # Clean up
        print("\nCleaning up...")
        hand.end()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        try:
            hand.end()
        except:
            pass
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        try:
            hand.end()
        except:
            pass
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
