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
        choices=["basic", "pose", "force", "full", "tactile"],
        default="basic",
        help="Demo to run (default: basic)"
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
