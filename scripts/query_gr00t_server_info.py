#!/usr/bin/env python3
"""Ask a GR00T PolicyServer for lightweight metadata.

This intentionally avoids sending images or robot state; it only calls RPCs
that should describe the server or prove that the endpoint is alive.
"""
from __future__ import annotations

import argparse
import pprint
import time

import msgpack
import msgpack_numpy
import zmq


msgpack_numpy.patch()


def roundtrip(sock, endpoint, data=None):
    request = {"endpoint": endpoint}
    if data is not None:
        request["data"] = data
    payload = msgpack.packb(
        request, default=msgpack_numpy.encode, use_bin_type=True)
    t0 = time.time()
    sock.send(payload)
    reply = msgpack.unpackb(sock.recv(), raw=False)
    return reply, (time.time() - t0) * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("server_url")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.SNDTIMEO, int(args.timeout_s * 1000))
    sock.setsockopt(zmq.RCVTIMEO, int(args.timeout_s * 1000))
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(args.server_url)

    try:
        for endpoint in ("ping", "get_modality_config"):
            print(f"\n=== {endpoint} ===")
            try:
                reply, rtt_ms = roundtrip(sock, endpoint)
                print(f"RTT: {rtt_ms:.1f} ms")
                pprint.pp(reply, width=120, compact=False)
            except Exception as exc:
                print(f"FAILED: {type(exc).__name__}: {exc}")
                sock.close(linger=0)
                sock = ctx.socket(zmq.REQ)
                sock.setsockopt(zmq.SNDTIMEO, int(args.timeout_s * 1000))
                sock.setsockopt(zmq.RCVTIMEO, int(args.timeout_s * 1000))
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(args.server_url)
    finally:
        sock.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()
