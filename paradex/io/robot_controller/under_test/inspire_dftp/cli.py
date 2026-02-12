#!/usr/bin/env python3
"""
Command-line interface for controlling the Inspire Hand.
"""

import argparse
import sys
import time
from typing import List, Optional

from hand_rh56dftp import InspireHandRH56DFTP as InspireHand, FingerID


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    
    parser = argparse.ArgumentParser(
        description="Control the Inspire Hand RH56dfq robotic hand."
    )
    
    # Connection options
    parser.add_argument(
        "--port", "-p",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port to connect to (default: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--baudrate", "-b",
        type=int,
        default=115200,
        help="Baudrate for serial connection (default: 115200)"
    )
    parser.add_argument(
        "--slave-id", "-s",
        type=int,
        default=1,
        help="Modbus slave ID (default: 1)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug output"
    )
    
    # Subparsers for different command groups
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get information about the hand")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset the hand")
    
    # Open command
    open_parser = subparsers.add_parser("open", help="Open fingers")
    open_parser.add_argument(
        "finger",
        nargs="?",
        type=str,
        default="all",
        help="Finger to open (all, little, ring, middle, index, thumb_bend, thumb_rotate)"
    )
    
    # Close command
    close_parser = subparsers.add_parser("close", help="Close fingers")
    close_parser.add_argument(
        "finger",
        nargs="?",
        type=str,
        default="all",
        help="Finger to close (all, little, ring, middle, index, thumb_bend, thumb_rotate)"
    )
    
    # Move command
    move_parser = subparsers.add_parser("move", help="Move a finger to a specific angle")
    move_parser.add_argument(
        "finger",
        type=str,
        help="Finger to move (little, ring, middle, index, thumb_bend, thumb_rotate)"
    )
    move_parser.add_argument(
        "angle",
        type=int,
        help="Angle to move to (0-1000, 0=closed, 1000=open)"
    )
    
    # Speed command
    speed_parser = subparsers.add_parser("speed", help="Set finger speed")
    speed_parser.add_argument(
        "finger",
        type=str,
        help="Finger to set speed for (all, little, ring, middle, index, thumb_bend, thumb_rotate)"
    )
    speed_parser.add_argument(
        "speed",
        type=int,
        help="Speed value (0-1000)"
    )
    
    # Force command
    force_parser = subparsers.add_parser("force", help="Set finger force threshold")
    force_parser.add_argument(
        "finger",
        type=str,
        help="Finger to set force threshold for (all, little, ring, middle, index, thumb_bend, thumb_rotate)"
    )
    force_parser.add_argument(
        "force",
        type=int,
        help="Force threshold value (0-1000)"
    )
    
    # Gesture command
    gesture_parser = subparsers.add_parser("gesture", help="Perform a predefined gesture")
    gesture_parser.add_argument(
        "gesture",
        type=str,
        choices=["pinch", "point", "thumbs_up", "grip"],
        help="Gesture to perform"
    )
    gesture_parser.add_argument(
        "--force", "-f",
        type=int,
        default=500,
        help="Force threshold for gestures that use force (default: 500)"
    )
    
    # Calibrate command
    calibrate_parser = subparsers.add_parser(
        "calibrate", 
        help="Calibrate force sensors (ensure hand is not touching anything during calibration)"
    )
    
    # Save command
    save_parser = subparsers.add_parser("save", help="Save settings to flash memory")
    
    # Factory reset command
    factory_reset_parser = subparsers.add_parser(
        "factory-reset", 
        help="Restore factory default settings"
    )
    
    # Interactive mode
    interactive_parser = subparsers.add_parser(
        "interactive", 
        help="Start interactive control mode"
    )
    
    return parser


def get_finger_by_name(hand: InspireHand, finger_name: str):
    """Get a finger object by its name."""
    finger_name = finger_name.lower()
    
    if finger_name == "little":
        return hand.little_finger
    elif finger_name == "ring":
        return hand.ring_finger
    elif finger_name == "middle":
        return hand.middle_finger
    elif finger_name == "index":
        return hand.index_finger
    elif finger_name == "thumb_bend":
        return hand.thumb_bend
    elif finger_name == "thumb_rotate":
        return hand.thumb_rotate
    elif finger_name == "all":
        return None
    else:
        raise ValueError(f"Unknown finger name: {finger_name}")


def print_finger_status(finger, header=None):
    """Print the status of a finger."""
    if header:
        print(f"\n{header}")
    
    print(f"  Angle: {finger.angle}")
    print(f"  Force: {finger.force}")
    print(f"  Status: {finger.status.name}")
    print(f"  Error: {finger.error}")
    print(f"  Temperature: {finger.temperature}")


def print_hand_status(hand: InspireHand):
    """Print the status of all fingers."""
    print("\nHand Status:")
    
    for finger in hand.fingers:
        print_finger_status(finger, f"{finger.name.title()} Finger:")


def interactive_mode(hand: InspireHand):
    """Start interactive control mode."""
    print("Inspire Hand Interactive Control Mode")
    print("Type 'help' for a list of commands, 'exit' to quit")
    
    while True:
        try:
            cmd_input = input("\n> ").strip().lower()
            
            if cmd_input in ("exit", "quit", "q"):
                break
                
            elif cmd_input in ("help", "h", "?"):
                print("\nAvailable commands:")
                print("  status              - Show status of all fingers")
                print("  open [finger]       - Open a finger or all fingers")
                print("  close [finger]      - Close a finger or all fingers")
                print("  move <finger> <ang> - Move finger to angle (0-1000)")
                print("  speed <finger> <s>  - Set finger speed (0-1000)")
                print("  force <finger> <f>  - Set finger force threshold (0-1000)")
                print("  pinch [force]       - Perform pinch gesture")
                print("  point               - Perform point gesture")
                print("  thumbs_up           - Perform thumbs-up gesture")
                print("  grip [force]        - Perform grip gesture")
                print("  reset               - Reset the hand")
                print("  calibrate           - Calibrate force sensors")
                print("  save                - Save settings to flash memory")
                print("  factory-reset       - Restore factory defaults")
                print("  exit                - Exit interactive mode")
                
            elif cmd_input == "status":
                print_hand_status(hand)
                
            elif cmd_input.startswith("open"):
                parts = cmd_input.split(maxsplit=1)
                finger_name = parts[1] if len(parts) > 1 else "all"
                
                if finger_name == "all":
                    hand.open_all_fingers()
                    print("Opened all fingers")
                else:
                    finger = get_finger_by_name(hand, finger_name)
                    finger.open()
                    print(f"Opened {finger_name} finger")
                    
            elif cmd_input.startswith("close"):
                parts = cmd_input.split(maxsplit=1)
                finger_name = parts[1] if len(parts) > 1 else "all"
                
                if finger_name == "all":
                    hand.close_all_fingers()
                    print("Closed all fingers")
                else:
                    finger = get_finger_by_name(hand, finger_name)
                    finger.close()
                    print(f"Closed {finger_name} finger")
                    
            elif cmd_input.startswith("move"):
                parts = cmd_input.split()
                if len(parts) != 3:
                    print("Usage: move <finger> <angle>")
                    continue
                    
                finger_name = parts[1]
                try:
                    angle = int(parts[2])
                except ValueError:
                    print("Angle must be an integer")
                    continue
                    
                if not 0 <= angle <= 1000:
                    print("Angle must be between 0 and 1000")
                    continue
                    
                finger = get_finger_by_name(hand, finger_name)
                finger.move(angle)
                print(f"Moved {finger_name} finger to angle {angle}")
                
            elif cmd_input.startswith("speed"):
                parts = cmd_input.split()
                if len(parts) != 3:
                    print("Usage: speed <finger> <speed>")
                    continue
                    
                finger_name = parts[1]
                try:
                    speed = int(parts[2])
                except ValueError:
                    print("Speed must be an integer")
                    continue
                    
                if not 0 <= speed <= 1000:
                    print("Speed must be between 0 and 1000")
                    continue
                    
                if finger_name == "all":
                    hand.set_all_finger_speeds(speed)
                    print(f"Set speed of all fingers to {speed}")
                else:
                    finger_id = FingerID.__members__[finger_name.upper()].value
                    hand.set_finger_speed(finger_id, speed)
                    print(f"Set speed of {finger_name} finger to {speed}")
                    
            elif cmd_input.startswith("force"):
                parts = cmd_input.split()
                if len(parts) != 3:
                    print("Usage: force <finger> <force>")
                    continue
                    
                finger_name = parts[1]
                try:
                    force = int(parts[2])
                except ValueError:
                    print("Force must be an integer")
                    continue
                    
                if not 0 <= force <= 1000:
                    print("Force must be between 0 and 1000")
                    continue
                    
                if finger_name == "all":
                    hand.set_all_finger_forces(force)
                    print(f"Set force threshold of all fingers to {force}")
                else:
                    finger_id = FingerID.__members__[finger_name.upper()].value
                    hand.set_finger_force(finger_id, force)
                    print(f"Set force threshold of {finger_name} finger to {force}")
                    
            elif cmd_input.startswith("pinch"):
                parts = cmd_input.split()
                force = 500
                if len(parts) > 1:
                    try:
                        force = int(parts[1])
                    except ValueError:
                        print("Force must be an integer")
                        continue
                        
                    if not 0 <= force <= 1000:
                        print("Force must be between 0 and 1000")
                        continue
                        
                hand.pinch(force)
                print(f"Performed pinch gesture with force {force}")
                
            elif cmd_input == "point":
                hand.point()
                print("Performed point gesture")
                
            elif cmd_input == "thumbs_up":
                hand.thumbs_up()
                print("Performed thumbs-up gesture")
                
            elif cmd_input.startswith("grip"):
                parts = cmd_input.split()
                force = 500
                if len(parts) > 1:
                    try:
                        force = int(parts[1])
                    except ValueError:
                        print("Force must be an integer")
                        continue
                        
                    if not 0 <= force <= 1000:
                        print("Force must be between 0 and 1000")
                        continue
                        
                hand.grip(force)
                print(f"Performed grip gesture with force {force}")
                
            elif cmd_input == "reset":
                hand.reset()
                print("Reset hand")
                
            elif cmd_input == "calibrate":
                print("Calibrating force sensors (do not touch the hand)...")
                hand.calibrate_force_sensors()
                print("Calibration complete")
                
            elif cmd_input == "save":
                hand.save_settings()
                print("Saved settings to flash memory")
                
            elif cmd_input == "factory-reset":
                hand.restore_factory_defaults()
                print("Restored factory defaults")
                
            else:
                print(f"Unknown command: {cmd_input}")
                print("Type 'help' for a list of commands")
                
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error: {e}")


def main(args: Optional[List[str]] = None):
    """Main entry point for the command-line interface."""
    
    parser = create_parser()
    args = parser.parse_args(args)
    
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        # Connect to hand
        hand = InspireHand(
            port=args.port,
            baudrate=args.baudrate,
            slave_id=args.slave_id,
            debug=args.debug
        )
        
        hand.open()
        print(f"Connected to Inspire Hand on {args.port}")
        
        # Execute command
        if args.command == "info":
            print_hand_status(hand)
            
        elif args.command == "reset":
            hand.reset()
            print("Reset hand")
            
        elif args.command == "open":
            if args.finger == "all":
                hand.open_all_fingers()
                print("Opened all fingers")
            else:
                finger = get_finger_by_name(hand, args.finger)
                finger.open()
                print(f"Opened {args.finger} finger")
                
        elif args.command == "close":
            if args.finger == "all":
                hand.close_all_fingers()
                print("Closed all fingers")
            else:
                finger = get_finger_by_name(hand, args.finger)
                finger.close()
                print(f"Closed {args.finger} finger")
                
        elif args.command == "move":
            finger = get_finger_by_name(hand, args.finger)
            finger.move(args.angle)
            print(f"Moved {args.finger} finger to angle {args.angle}")
            
        elif args.command == "speed":
            if args.finger == "all":
                hand.set_all_finger_speeds(args.speed)
                print(f"Set speed of all fingers to {args.speed}")
            else:
                finger_id = FingerID.__members__[args.finger.upper()].value
                hand.set_finger_speed(finger_id, args.speed)
                print(f"Set speed of {args.finger} finger to {args.speed}")
                
        elif args.command == "force":
            if args.finger == "all":
                hand.set_all_finger_forces(args.force)
                print(f"Set force threshold of all fingers to {args.force}")
            else:
                finger_id = FingerID.__members__[args.finger.upper()].value
                hand.set_finger_force(finger_id, args.force)
                print(f"Set force threshold of {args.finger} finger to {args.force}")
                
        elif args.command == "gesture":
            if args.gesture == "pinch":
                hand.pinch(args.force)
                print(f"Performed pinch gesture with force {args.force}")
            elif args.gesture == "point":
                hand.point()
                print("Performed point gesture")
            elif args.gesture == "thumbs_up":
                hand.thumbs_up()
                print("Performed thumbs-up gesture")
            elif args.gesture == "grip":
                hand.grip(args.force)
                print(f"Performed grip gesture with force {args.force}")
                
        elif args.command == "calibrate":
            print("Calibrating force sensors (do not touch the hand)...")
            hand.calibrate_force_sensors()
            print("Calibration complete")
            
        elif args.command == "save":
            hand.save_settings()
            print("Saved settings to flash memory")
            
        elif args.command == "factory-reset":
            hand.restore_factory_defaults()
            print("Restored factory defaults")
            
        elif args.command == "interactive":
            interactive_mode(hand)
            
        else:
            print(f"Unknown command: {args.command}")
            return 1
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    finally:
        # Disconnect from hand
        try:
            if 'hand' in locals():
                hand.close()
                print("Disconnected from Inspire Hand")
        except Exception:
            pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 