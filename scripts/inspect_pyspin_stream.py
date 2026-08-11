#!/usr/bin/env python3
"""
Inspect PySpin camera stream/node state and attempt to start acquisition.

Usage:
  python scripts/inspect_pyspin_stream.py

This script will:
 - Load cameras listed in system/current/pc.json
 - Print key node values (TriggerMode, TriggerSource, PixelFormat, StreamEnable/related nodes if present)
 - Attempt to start continuous acquisition and capture one frame, printing full traceback on failure

Run on the capture PC. If another process (server_daemon.py) is running, stop it first.
"""

import traceback
import sys
from paradex.utils.system import get_camera_list
from paradex.io.camera_system.pyspin import load_camera


def safe_node_get(cam, name, ntype):
    try:
        return cam._get_node(cam.nodeMap, name, ntype, readable=True, writable=False)
    except Exception:
        return None


def print_node(cam, name, ntype):
    node = safe_node_get(cam, name, ntype)
    if node is None:
        print(f"  {name}: <missing>")
        return
    try:
        if ntype == 'enum':
            cur = node.GetCurrentEntry().GetSymbolic()
            print(f"  {name}: {cur}")
        else:
            print(f"  {name}: {node.GetValue()}")
    except Exception as e:
        print(f"  {name}: <error reading: {e}>")


def inspect_camera(serial):
    print(f"\n--- Inspecting camera {serial} ---")
    try:
        cam = load_camera(serial)
    except Exception as e:
        print(f"Failed to load camera {serial}: {e}")
        return

    # print some useful nodes
    nodes = [
        ('TriggerMode', 'enum'),
        ('TriggerSource', 'enum'),
        ('TriggerSelector', 'enum'),
        ('TriggerOverlap', 'enum'),
        ('PixelFormat', 'enum'),
        ('GainAuto', 'enum'),
        ('Gain', 'float'),
        ('ExposureAuto', 'enum'),
        ('ExposureTime', 'float'),
        ('BalanceWhiteAuto', 'enum'),
        ('BalanceRatio', 'float'),
        ('Gamma', 'float'),
        ('GevSCPSPacketSize', 'int'),
    ]

    print('Node values:')
    for name, ntype in nodes:
        print_node(cam, name, ntype)

    # Attempt to start continuous acquisition
    print('\nAttempting to start continuous acquisition...')
    try:
        cam.start('continuous', False, frame_rate=30)
        print('cam.start() succeeded, trying to get one frame...')
        frame, meta = cam.get_image()
        if frame is None:
            print('get_image returned None')
        else:
            print(f'Got frame id {meta.get("frameID")} size {frame.shape}')
        cam.stop()
    except Exception as e:
        print('Exception while starting/capturing:')
        traceback.print_exc()

    try:
        cam.release()
    except Exception:
        pass


def main():
    serials = get_camera_list()
    if not serials:
        print('No cameras configured in system/current/pc.json')
        return

    print('Configured cameras:', serials)
    for s in serials:
        inspect_camera(s)


if __name__ == '__main__':
    main()
