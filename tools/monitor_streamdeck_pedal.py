#!/usr/bin/env python3
"""Print Stream Deck Pedal press and release events in real time."""

import argparse
import sys
import threading
from datetime import datetime

from StreamDeck.DeviceManager import DeviceManager


PEDAL_NAMES = ("LEFT", "MIDDLE", "RIGHT")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print Stream Deck Pedal input events until Ctrl+C is pressed."
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Pedal device index to monitor when multiple devices are connected (default: 0)",
    )
    parser.add_argument(
        "--poll-hz",
        type=int,
        default=50,
        help="USB polling frequency from 1 to 1000 Hz (default: 50)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not 1 <= args.poll_hz <= 1000:
        print("--poll-hz must be between 1 and 1000", file=sys.stderr)
        return 2

    devices = [
        device
        for device in DeviceManager().enumerate()
        if device.deck_type() == "Stream Deck Pedal"
    ]

    if not devices:
        print(
            "Stream Deck Pedal not found (expected USB VID:PID 0fd9:0086).",
            file=sys.stderr,
        )
        return 1

    if not 0 <= args.device_index < len(devices):
        print(
            "--device-index {} is invalid; found {} pedal device(s)".format(
                args.device_index, len(devices)
            ),
            file=sys.stderr,
        )
        return 2

    pedal = devices[args.device_index]
    states = [False] * len(PEDAL_NAMES)
    stopped = threading.Event()

    def on_key_change(_deck, key, pressed):
        if not 0 <= key < len(states):
            print("Unknown pedal index: {}".format(key), flush=True)
            return

        states[key] = bool(pressed)
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        state_text = " ".join("1" if state else "0" for state in states)
        event = "DOWN" if pressed else "UP"
        print(
            "{}  {:<6} {:<4}  state=[{}]".format(
                timestamp, PEDAL_NAMES[key], event, state_text
            ),
            flush=True,
        )

    try:
        pedal.open()
        pedal.set_poll_frequency(args.poll_hz)
        pedal.set_key_callback(on_key_change)

        print("Stream Deck Pedal connected.")
        print("Mapping: 0=LEFT, 1=MIDDLE, 2=RIGHT")
        print("Press or release a pedal; Ctrl+C exits.", flush=True)

        while not stopped.wait(0.25):
            if not pedal.connected():
                print("Stream Deck Pedal disconnected.", file=sys.stderr)
                return 1
    except KeyboardInterrupt:
        print("\nStopping pedal monitor.")
    finally:
        if pedal.is_open():
            pedal.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
