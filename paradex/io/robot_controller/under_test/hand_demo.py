#!/usr/bin/env python3
"""
Demo script for the Inspire Hand RH56DFTP.
Demonstrates connection, pose reading/writing, and contact information reading.
"""

import sys
import time
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hand_rh56dftp import InspireHandRH56DFTP, ContactStatus


def parse_args():
    parser = argparse.ArgumentParser(description="Inspire Hand RH56DFTP Demo")
    parser.add_argument(
        "--port", "-p",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port for the Inspire Hand (default: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--baudrate", "-b",
        type=int,
        default=115200,
        help="Baudrate for the serial connection (default: 115200)"
    )
    parser.add_argument(
        "--slave-id", "-s",
        type=int,
        default=1,
        help="Modbus slave ID (default: 1)"
    )
    parser.add_argument(
        "--demo", "-d",
        type=str,
        choices=["basic", "pose", "contact", "full", "tactile"],
        default="basic",
        help="Demo to run (default: basic)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    return parser.parse_args()


def print_pose(hand: InspireHandRH56DFTP):
    """Print the current pose of the hand."""
    try:
        pose = hand.read_pose()
        angles = hand.read_angles()
        
        print("\n=== Current Pose ===")
        print(f"Pose values: {pose}")
        print(f"Angle values: {angles}")
        
        finger_names = ["Little", "Ring", "Middle", "Index", "Thumb Bend", "Thumb Rotate"]
        for i, name in enumerate(finger_names):
            print(f"  {name}: pose={pose[i]}, angle={angles[i]}")
    except Exception as e:
        print(f"Error reading pose: {e}")


def print_contact_info(hand: InspireHandRH56DFTP):
    """Print contact information from the hand."""
    try:
        contact_info = hand.read_contact_info()
        
        print("\n=== Contact Information ===")
        print(f"Contact status: {contact_info}")
        
        finger_names = ["Little", "Ring", "Middle", "Index", "Thumb Bend", "Thumb Rotate"]
        for i, name in enumerate(finger_names):
            in_contact = contact_info.is_contact(i)
            force = contact_info.get_contact_force(i)
            print(f"  {name}: contact={in_contact}, force={force}")
    except Exception as e:
        print(f"Error reading contact info: {e}")


def basic_demo(hand: InspireHandRH56DFTP):
    """Run a basic demo showing connection, pose reading, and pose writing."""
    print("\n=== Basic Demo ===")
    
    # Read current pose
    print("\n1. Reading current pose...")
    print_pose(hand)
    
    # Open all fingers (set all angles to 0)
    
    print("\n2. Opening all fingers (pose = [0, 0, 0, 0, 0, 0])...")
    hand.write_pose([0, 0, 0, 0, 0, 0])
    time.sleep(2)
    
    # Read pose again
    print("\n3. Reading pose after opening...")
    print_pose(hand)
    
    # Close all fingers (set all angles to 0)
    print("\n4. Closing all fingers (pose = [0, 0, 0, 0, 0, 0])...")
    hand.write_pose([0, 0, 0, 0, 0, 0])
    time.sleep(2)
    
    # Read pose again
    print("\n5. Reading pose after closing...")
    print_pose(hand)
    
    # Set to middle position
    print("\n6. Setting to middle position (pose = [500, 500, 500, 500, 500, 500])...")
    hand.write_pose([500, 500, 500, 500, 500, 500])
    time.sleep(2)
    
    # Read final pose
    print("\n7. Final pose:")
    print_pose(hand)


def pose_demo(hand: InspireHandRH56DFTP):
    """Demonstrate pose reading and writing."""
    print("\n=== Pose Demo ===")
    
    # Read initial pose
    print("\nInitial pose:")
    print_pose(hand)
    
    # Try different poses
    poses = [
        ([0, 0, 0, 0, 0, 0], "Fully open"),
        ([1500, 1500, 1500, 1500, 1500, 1500], "Closed"),
        ([500, 500, 500, 500, 500, 500], "Middle position"),
        ([999, 0, 999, 0, 999, 500], "Alternating pattern"),
    ]
    
    for pose, description in poses:
        print(f"\nSetting pose: {description}")
        print(f"  Values: {pose}")
        hand.write_pose(pose)
        time.sleep(2)
        
        print("Current pose after setting:")
        print_pose(hand)
        time.sleep(1)


def contact_demo(hand: InspireHandRH56DFTP):
    """Demonstrate contact information reading."""
    print("\n=== Contact Information Demo ===")
    
    print("\n1. Reading contact information (hand should be open, no contact)...")
    print_contact_info(hand)
    
    print("\n2. Closing fingers to create contact...")
    hand.write_pose([0, 0, 0, 0, 0, 0])
    time.sleep(2)
    
    print("\n3. Reading contact information after closing...")
    print_contact_info(hand)
    
    print("\n4. Opening fingers...")
    hand.write_pose([999, 999, 999, 999, 999, 999])
    time.sleep(2)
    
    print("\n5. Reading contact information after opening...")
    print_contact_info(hand)
    
    # Monitor contact in real-time
    print("\n6. Monitoring contact information (press Ctrl+C to stop)...")
    try:
        for i in range(300):
            print(f"\n--- Reading {i+1}/10 ---")
            print_contact_info(hand)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")


def full_demo(hand: InspireHandRH56DFTP):
    """Run a comprehensive demo showing all features."""
    print("\n=== Full Demo ===")
    
    # Reset hand
    print("\n1. Resetting hand...")
    try:
        hand.reset()
        time.sleep(1)
    except Exception as e:
        print(f"   Note: {e}")
    
    # Set speed
    print("\n2. Setting speed to 1000...")
    try:
        hand.set_speed(1000)
    except Exception as e:
        print(f"   Note: {e}")
    
    # Set force threshold
    print("\n3. Setting force threshold to 300...")
    try:
        hand.set_force_threshold(300)
    except Exception as e:
        print(f"   Note: {e}")
    
    # Read initial status
    print("\n4. Reading initial status...")
    try:
        status = hand.get_status()
        print(f"   Status keys: {list(status.keys())}")
    except Exception as e:
        print(f"   Note: {e}")
    
    # Pose operations
    print("\n5. Pose operations...")
    print_pose(hand)
    
    # Contact operations
    print("\n6. Contact operations...")
    print_contact_info(hand)
    
    print("\n7. Angle operations...")
    for idx in range(6):
        hand.write_angles([1000, 1000, 1000, 1000, 1000, 1000])
        print(f"\n   Moving joint {idx} to close...")
        for joint_angle in range(0, 1000, 200):
            hand.write_angle_by_id(idx, joint_angle)
            time.sleep(0.1)
            print(f"   Current angle:")
            angle_values = hand.read_angles()
            print(f"     {angle_values}")

    # Move through a sequence
    print("\n8. Moving through a sequence...")
    sequence = [
        ([2000, 2000, 2000, 2000, 2000, 2000], "Fully Closed"),
        ([500, 500, 500, 500, 500, 500], "Middle Position"),
        ([0, 0, 0, 0, 0, 0], "Fully Open"),
        ([999, 500, 0, 500, 999, 500], "Alternating Pattern"),
    ]
    
    for pose, name in sequence:
        print(f"\n   Moving to: {name}")
        hand.write_pose(pose)
        time.sleep(2)
        
        print("   Current pose:")
        pose_values = hand.read_pose()
        print(f"     {pose_values}")
        
        print("   Contact info:")
        contact_info = hand.read_contact_info()
        contacts = contact_info.get_all_contacts()
        print(f"     {contacts}")

    # Fully Open all fingers
    for idx in range(6):
        hand.write_pose([0, 0, 0, 0, 0, 0])
        print(f"\n   Moving joint {idx} to close...")
        for joint_pose in range(0, 2000, 400):
            hand.write_pose_by_id(idx, joint_pose)
            time.sleep(0.1)
            print(f"   Current pose:")
            pose_values = hand.read_pose()
            print(f"     {pose_values}")


def tactile_demo(hand: InspireHandRH56DFTP):
    """Demonstrate tactile information reading with visualization."""
    print("\n=== Tactile Information Demo ===")
    print("\n1. Opening hand...")
    hand.write_pose([0, 0, 0, 0, 0, 0])
    
    # Sensor shapes (rows, cols)
    SENSOR_SHAPES = {
        'little': {'tip': (3, 3), 'nail': (12, 8), 'pad': (10, 8)},
        'ring':   {'tip': (3, 3), 'nail': (12, 8), 'pad': (10, 8)},
        'middle': {'tip': (3, 3), 'nail': (12, 8), 'pad': (10, 8)},
        'index':  {'tip': (3, 3), 'nail': (12, 8), 'pad': (10, 8)},
        'thumb':  {'tip': (3, 3), 'nail': (12, 8), 'mid': (3, 3), 'pad': (12, 8)},
        'palm':   (8, 14)
    }
    
    # Setup plot
    plt.ion()
    fig, axs = plt.subplots(5, 4, figsize=(16, 20))
    axs = axs.flatten()
    
    # Map plot index to sensor part
    # We have 5 fingers * 3 parts + thumb extra + palm = 15 + 1 + 1 = 17 parts?
    # Let's flatten the structure to map to subplots
    plot_mapping = []
    
    fingers = ['little', 'ring', 'middle', 'index', 'thumb']
    for finger in fingers:
        parts = ['tip', 'nail', 'pad']
        if finger == 'thumb':
            parts = ['tip', 'nail', 'mid', 'pad']
            
        for part in parts:
            plot_mapping.append((finger, part))
            
    plot_mapping.append(('palm', 'palm'))
    
    # Initialize images
    im_list = []
    for i, (finger, part) in enumerate(plot_mapping):
        if i >= len(axs): break
        
        ax = axs[i]
        
        if finger == 'palm':
            rows, cols = SENSOR_SHAPES['palm']
            # Palm is transposed in visualization logic
            shape = (cols, rows) 
        else:
            shape = SENSOR_SHAPES[finger][part]
            
        dummy = np.zeros(shape)
        im = ax.imshow(dummy, cmap='viridis', vmin=0, vmax=4095)
        ax.set_title(f"{finger} {part}")
        plt.colorbar(im, ax=ax, shrink=0.8)
        im_list.append(im)
        
    plt.tight_layout()
    
    print("Starting visualization loop. Press Ctrl+C to stop.")
    try:
        while True:
            data = hand.read_tactile_data()
            
            for i, (finger, part) in enumerate(plot_mapping):
                if i >= len(im_list): break
                
                if finger == 'palm':
                    raw_data = data['palm']
                    rows, cols = SENSOR_SHAPES['palm']
                    # Palm logic from visualize_contact.py:
                    # array = np.array(raw, dtype=np.uint16).reshape(rows, cols)
                    # array = array[::-1].T
                    array = np.array(raw_data, dtype=np.uint32).reshape(rows, cols)
                    array = array[::-1].T
                else:
                    raw_data = data[finger][part]
                    rows, cols = SENSOR_SHAPES[finger][part]
                    array = np.array(raw_data, dtype=np.uint32).reshape(rows, cols)
                
                im_list[i].set_data(array)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            # plt.pause(0.01) # pause can be slow, flush_events is often better for loops
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
        plt.close(fig)




def main():
    """Main demo function."""
    args = parse_args()
    
    try:
        # Create hand instance
        hand = InspireHandRH56DFTP(
            port=args.port,
            baudrate=args.baudrate,
            slave_id=args.slave_id,
            debug=args.debug
        )
        
        # Connect to hand
        print(f"Connecting to Inspire Hand RH56DFTP on {args.port}...")
        with hand.connect():
            print("Connected successfully!")
            
            # Run the selected demo
            if args.demo == "basic":
                basic_demo(hand)
            elif args.demo == "pose":
                pose_demo(hand)
            elif args.demo == "contact":
                contact_demo(hand)
            elif args.demo == "full":
                full_demo(hand)
            elif args.demo == "tactile":
                tactile_demo(hand)
            print("\n=== Demo completed successfully! ===")
            
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


