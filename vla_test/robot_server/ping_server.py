"""Minimal connectivity ping for policy_server.py.

Sends one dummy request matching the server's expected schema and prints the
round-trip time. Use this from the robot-pc to verify the ssh tunnel + ZMQ
wire is alive, without copying any LeRobot data over.

Usage:
  python scripts/deploy/ping_server.py --server tcp://localhost:5600
"""

import argparse
import time

import msgpack
import msgpack_numpy
import numpy as np
import zmq

msgpack_numpy.patch()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="tcp://localhost:5600")
    args = p.parse_args()

    request = {
        "state.eef_position":       np.zeros(3, dtype=np.float64),
        "state.eef_rotation":       np.zeros(3, dtype=np.float64),
        "state.arm_joint_position": np.zeros(6, dtype=np.float64),
        "state.dexhand_position":   np.zeros(6, dtype=np.float64),
        "video.front":              np.zeros((480, 640, 3), dtype=np.uint8),
        "language.instruction":     "ping",
    }

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(args.server)
    print(f"Connected to {args.server}")

    t0 = time.time()
    sock.send(msgpack.packb(request, default=msgpack_numpy.encode, use_bin_type=True))
    reply = msgpack.unpackb(sock.recv(), raw=False)
    rtt_ms = (time.time() - t0) * 1000

    print(f"RTT={rtt_ms:.1f}ms  server={reply['elapsed_ms']:.1f}ms  "
          f"network={rtt_ms - reply['elapsed_ms']:.1f}ms")
    print(f"chunk shapes: pos={reply['action.eef_position'].shape} "
          f"rot={reply['action.eef_rotation'].shape} "
          f"dex={reply['action.dexhand_position'].shape}")


if __name__ == "__main__":
    main()
