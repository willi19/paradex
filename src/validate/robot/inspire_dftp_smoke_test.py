#!/usr/bin/env python3
"""Bimanual Modbus TCP connection and motion test for Inspire RH56DFTP."""

from __future__ import annotations

import argparse
import socket
import struct
import subprocess
import sys
import time

from pymodbus.client.sync import ModbusTcpClient


ANGLE_SET = 1486
ANGLE_ACT = 1546
JOINT_COUNT = 6
NO_ACTION = 0xFFFF
IP_UNICAST_IF = 50
JOINT_NAMES = (
    "little",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotate",
)


def response_registers(response, expected_count: int) -> list[int]:
    if response is None or response.isError():
        raise RuntimeError(f"Modbus read failed: {response}")
    registers = getattr(response, "registers", None)
    if registers is None or len(registers) != expected_count:
        raise RuntimeError(f"invalid Modbus response: {registers}")
    return [int(value) for value in registers]


def read_angles(client: ModbusTcpClient) -> list[int]:
    angles = response_registers(
        client.read_holding_registers(ANGLE_ACT, JOINT_COUNT),
        JOINT_COUNT,
    )
    if any(angle < 0 or angle > 2000 for angle in angles):
        raise RuntimeError(f"angle response outside expected 0..2000 range: {angles}")
    return angles


def write_one_joint(client: ModbusTcpClient, joint: int, target: int) -> None:
    targets = [NO_ACTION] * JOINT_COUNT
    targets[joint] = target
    response = client.write_registers(ANGLE_SET, targets)
    if response is None or response.isError():
        raise RuntimeError(f"Modbus write failed: {response}")


def nearby_target(current: int, delta: int) -> int:
    """Choose a nearby target, preferring a small closing movement."""
    if current >= delta:
        return current - delta
    return current + delta


def connect_client(
    client: ModbusTcpClient,
    ip: str,
    port: int,
    timeout: float,
    interface: str,
) -> None:
    interface_index = socket.if_nametoindex(interface)
    bound_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bound_socket.settimeout(timeout)
    try:
        bound_socket.setsockopt(
            socket.IPPROTO_IP,
            IP_UNICAST_IF,
            struct.pack("!I", interface_index),
        )
        bound_socket.connect((ip, port))
    except Exception:
        bound_socket.close()
        raise
    client.socket = bound_socket


def verify_kernel_route(ip: str, interface: str) -> None:
    result = subprocess.run(
        ["ip", "route", "get", ip],
        check=True,
        capture_output=True,
        text=True,
    )
    route = result.stdout.strip()
    if f"dev {interface}" not in route:
        raise RuntimeError(
            f"{ip} currently routes through a different NIC: {route}. "
            f"Run: sudo ip route replace {ip}/32 dev {interface} "
            "src 192.168.11.4"
        )


def test_hand_motion(
    side: str,
    client: ModbusTcpClient,
    joint: int,
    delta: int,
    hold: float,
) -> None:
    angles = read_angles(client)
    original = angles[joint]
    target = nearby_target(original, delta)
    joint_name = JOINT_NAMES[joint]
    moved = False

    try:
        print(f"[{side}] moving {joint_name}: {original} -> {target}")
        write_one_joint(client, joint, target)
        moved = True
        time.sleep(hold)
        measured = read_angles(client)[joint]
        if measured == original:
            raise RuntimeError(
                f"{side} {joint_name} did not report any movement "
                f"(still {measured}); check power or fault state"
            )
        print(f"[{side}] measured {joint_name}: {measured}")
        print(f"[OK] {side} hand motion observed")
    finally:
        if moved:
            write_one_joint(client, joint, original)
            time.sleep(hold)
            restored = read_angles(client)[joint]
            print(
                f"[{side}] restored {joint_name}: "
                f"target={original}, measured={restored}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Connect both Inspire RH56DFTP hands, then move and restore the "
            "left hand followed by the right hand."
        )
    )
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--left-ip", default="192.168.11.211")
    parser.add_argument("--right-ip", default="192.168.11.210")
    parser.add_argument(
        "--left-interface",
        default="enp8s0f2",
        help="NIC connected to the left hand (default: enp8s0f2)",
    )
    parser.add_argument(
        "--right-interface",
        default="enp8s0f1",
        help="NIC connected to the right hand (default: enp8s0f1)",
    )
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="connect to and read both hands without moving them",
    )
    parser.add_argument(
        "--joint",
        type=int,
        choices=range(JOINT_COUNT),
        default=5,
        metavar="0..5",
        help="0=little, 1=ring, 2=middle, 3=index, 4=thumb bend, 5=thumb rotate",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=100,
        help="motion size in the hand angle scale (allowed: 1..100)",
    )
    parser.add_argument("--hold", type=float, default=0.5)
    args = parser.parse_args()

    if not 0 < args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 0 < args.timeout <= 10:
        parser.error("--timeout must be between 0 and 10 seconds")
    if not 1 <= args.delta <= 100:
        parser.error("--delta must be between 1 and 100")
    if not 0 <= args.hold <= 5:
        parser.error("--hold must be between 0 and 5 seconds")
    return args


def main() -> int:
    args = parse_args()
    hands = (
        ("left", args.left_ip, args.left_interface),
        ("right", args.right_ip, args.right_interface),
    )
    clients: dict[str, ModbusTcpClient] = {}

    try:
        for side, ip, interface in hands:
            verify_kernel_route(ip, interface)
            client = ModbusTcpClient(
                ip,
                port=args.port,
                timeout=args.timeout,
            )
            connect_client(
                client,
                ip,
                args.port,
                args.timeout,
                interface,
            )
            clients[side] = client
            angles = read_angles(client)
            print(
                f"[OK] {side} DFTP responded on "
                f"{ip}:{args.port} via {interface}"
            )
            print(f"[{side}] actual angles: {dict(zip(JOINT_NAMES, angles))}")

        if args.read_only:
            print("[OK] read-only bimanual connection check complete")
            return 0

        for side, _ip, _interface in hands:
            test_hand_motion(
                side,
                clients[side],
                args.joint,
                args.delta,
                args.hold,
            )
        print("[OK] left then right bimanual motion test complete")
        return 0
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    finally:
        for client in clients.values():
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
