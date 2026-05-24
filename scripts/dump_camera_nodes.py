#!/usr/bin/env python3
"""
Dump camera node current/min/max values for common nodes.

Usage:
  python3 scripts/dump_camera_nodes.py

Run on the capture PC. This will not start acquisition; it only reads node values.
"""

import traceback
from paradex.utils.system import get_camera_list
from paradex.io.camera_system.pyspin import load_camera


def node_info(cam, name, ntype):
    try:
        node = cam._get_node(cam.nodeMap, name, ntype, True, False)
    except Exception as e:
        return (False, str(e))

    try:
        if ntype in ('float', 'int'):
            cur = node.GetValue()
            mn = node.GetMin()
            mx = node.GetMax()
            return (True, f'cur={cur}, min={mn}, max={mx}')
        elif ntype == 'enum':
            cur = node.GetCurrentEntry().GetSymbolic()
            entries = [e.GetSymbolic() for e in node.GetEntries()]
            return (True, f'cur={cur}, entries={entries}')
        elif ntype == 'bool':
            cur = node.GetValue()
            return (True, f'cur={cur}')
        else:
            return (True, 'readable')
    except Exception as e:
        return (False, str(e))


def inspect(serial):
    print(f"\n--- Camera {serial} ---")
    try:
        cam = load_camera(serial)
    except Exception as e:
        print('Failed to load camera:', e)
        return

    nodes = [
        ('Gain', 'float'),
        ('GainAuto', 'enum'),
        ('ExposureTime', 'float'),
        ('ExposureAuto', 'enum'),
        ('Gamma', 'float'),
        ('BalanceWhiteAuto', 'enum'),
        ('PixelFormat', 'enum')
    ]

    for name, ntype in nodes:
        ok, info = node_info(cam, name, ntype)
        if ok:
            print(f'{name} ({ntype}): {info}')
        else:
            print(f'{name} ({ntype}): <unavailable or error: {info}>')

    try:
        cam.release()
    except Exception:
        pass


def main():
    serials = get_camera_list()
    if not serials:
        print('No cameras configured in system/current/pc.json')
        return

    for s in serials:
        try:
            inspect(s)
        except Exception:
            traceback.print_exc()


if __name__ == '__main__':
    main()
