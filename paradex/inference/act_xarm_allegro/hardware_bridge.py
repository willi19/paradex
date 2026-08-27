"""ROS-side xArm6 + Allegro v5 bridge.

Run this process from the ROS Humble/Python 3.8 environment.  It never loads
PyTorch or the policy. Live publication requires the explicit --allow-live
flag. A command-expiry watchdog can be enabled explicitly when desired.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Optional

import numpy as np
import zmq

from paradex.inference.act_xarm_allegro.transport import PROTOCOL_VERSION


class XArmAllegroHardware:
    def __init__(self, namespace: str, *, hand_slew_rate_rad_s: Optional[float]):
        from paradex.io.robot_controller.allegro_v5_controller_ros2 import AllegroController
        from paradex.io.robot_controller.xarm_controller_ros import XArmControllerROS

        self.arm = XArmControllerROS(namespace=namespace, servo_api="cartesian_aa")
        self.hand = AllegroController(namespace=namespace)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not self.hand.connection_event.is_set():
            time.sleep(0.01)
        feedback = self.feedback()
        if feedback[0].shape != (22,) or not feedback[2]:
            raise RuntimeError("Robot feedback did not become ready")
        self.hand.set_command_slew_rate(
            hand_slew_rate_rad_s,
            initial_action=feedback[0][6:],
        )

    def feedback(self) -> tuple[np.ndarray, np.ndarray, bool, bool, int]:
        arm = self.arm.get_data()
        hand = self.hand.get_data()
        arm_qpos = np.asarray(arm.get("qpos"), dtype=np.float64).reshape(-1)
        hand_qpos = np.asarray(hand.get("qpos"), dtype=np.float64).reshape(-1)
        tcp = np.asarray(arm.get("position"), dtype=np.float64)
        connected = bool(hand.get("is_connected"))
        valid = bool(
            arm_qpos.shape == (6,)
            and hand_qpos.shape == (16,)
            and tcp.shape == (4, 4)
            and np.all(np.isfinite(arm_qpos))
            and np.all(np.isfinite(hand_qpos))
            and np.all(np.isfinite(tcp))
        )
        state = np.concatenate((arm_qpos, hand_qpos)) if valid else np.full(22, np.nan)
        state_times = [
            arm.get("state_monotonic_time"),
            hand.get("state_monotonic_time"),
        ]
        finite_times = [float(value) for value in state_times if value is not None]
        state_monotonic_ns = (
            int(min(finite_times) * 1e9) if len(finite_times) == 2 else 0
        )
        error = bool(self.arm.is_error() or self.hand.is_error())
        return state, tcp, bool(connected and valid), bool(error), int(state_monotonic_ns)

    def target(self, tcp: np.ndarray, hand: np.ndarray) -> None:
        self.arm.move(np.asarray(tcp, dtype=np.float64))
        self.hand.move(np.asarray(hand, dtype=np.float64))

    def hand_target(self, hand: np.ndarray) -> None:
        self.hand.move(np.asarray(hand, dtype=np.float64))

    def hold_hand(self) -> None:
        state, _tcp, connected, _error, _timestamp = self.feedback()
        if connected and np.all(np.isfinite(state)):
            self.hand.move(state[6:])

    def hold(self) -> None:
        state, tcp, connected, _error, _timestamp = self.feedback()
        if connected and np.all(np.isfinite(state)) and np.all(np.isfinite(tcp)):
            self.arm.move(tcp)
            self.hand.move(state[6:])

    def close(self) -> None:
        try:
            self.hold()
        except Exception:
            pass
        try:
            self.hand.end()
        finally:
            self.arm.end()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACT xArm6 + Allegro v5 ROS hardware bridge")
    parser.add_argument("--state-endpoint", default="tcp://127.0.0.1:5561")
    parser.add_argument("--command-endpoint", default="tcp://127.0.0.1:5562")
    parser.add_argument("--namespace", default="right")
    parser.add_argument("--publish-hz", type=float, default=100.0)
    parser.add_argument(
        "--watchdog-ms",
        type=float,
        default=0.0,
        help="Optional command-expiry watchdog; disabled by default for direct ACT execution",
    )
    parser.add_argument(
        "--hand-slew-rate-rad-s",
        type=float,
        default=None,
        help="Optional Allegro output slew limit; disabled by default for direct ACT execution",
    )
    parser.add_argument("--allow-live", action="store_true")
    return parser.parse_args()


def _state_header(*, state_ns, connected, error, latched, allow_live, status):
    """Return a JSON-safe bridge header even when inputs are NumPy scalars."""

    return {
        "version": int(PROTOCOL_VERSION),
        "bridge_monotonic_ns": int(time.monotonic_ns()),
        "state_monotonic_ns": int(state_ns),
        "connected": bool(connected),
        "error": bool(error),
        "latched": bool(latched),
        "allow_live": bool(allow_live),
        "status": str(status),
    }


def main() -> None:
    args = parse_args()
    context = zmq.Context.instance()
    state_pub = context.socket(zmq.PUB)
    state_pub.bind(args.state_endpoint)
    command_pull = context.socket(zmq.PULL)
    command_pull.bind(args.command_endpoint)
    hardware = XArmAllegroHardware(
        args.namespace,
        hand_slew_rate_rad_s=args.hand_slew_rate_rad_s,
    )
    latched = False
    status = "ready"
    target_active = False
    target_scope = None
    last_valid_command_ns = 0
    period = 1.0 / args.publish_hz

    print(
        f"[act-hardware] state -> {args.state_endpoint}; commands <- {args.command_endpoint}; "
        f"allow_live={args.allow_live}"
    )
    try:
        while True:
            started = time.monotonic()
            latest = None
            while True:
                try:
                    latest = command_pull.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
            if latest is not None:
                kind = None
                try:
                    if len(latest) < 2 or latest[0] != b"command":
                        raise ValueError("invalid command packet")
                    header = json.loads(latest[1].decode("utf-8"))
                    if header.get("version") != PROTOCOL_VERSION:
                        raise ValueError("protocol version mismatch")
                    kind = header.get("kind")
                    issued_ns = int(header.get("issued_monotonic_ns", 0))
                    ttl_ms = float(header.get("ttl_ms", 0.0))
                    expired = (time.monotonic_ns() - issued_ns) / 1e6 > ttl_ms
                    if kind == "abort":
                        hardware.hold()
                        latched = True
                        target_active = False
                        target_scope = None
                        status = "aborted"
                    elif kind == "rearm":
                        _state, _tcp, connected, error, _stamp = hardware.feedback()
                        if connected and not error:
                            latched = False
                            status = "ready"
                    elif kind == "hold":
                        hardware.hold()
                        target_active = False
                        target_scope = None
                        status = "holding"
                    elif kind == "hand_hold":
                        if not args.allow_live:
                            raise PermissionError("bridge was not started with --allow-live")
                        hardware.hold_hand()
                        target_active = False
                        target_scope = None
                        status = "hand_holding"
                    elif kind == "hand_target":
                        if not args.allow_live:
                            raise PermissionError("bridge was not started with --allow-live")
                        if latched:
                            raise RuntimeError("bridge is latched")
                        if expired:
                            raise TimeoutError("hand target command expired before receipt")
                        if len(latest) != 3:
                            raise ValueError("hand target payload missing")
                        hand = np.frombuffer(latest[2], dtype=np.float64).reshape(16).copy()
                        if not np.all(np.isfinite(hand)):
                            raise ValueError("hand target contains non-finite values")
                        hardware.hand_target(hand)
                        target_active = True
                        target_scope = "hand"
                        last_valid_command_ns = time.monotonic_ns()
                        status = "commanding_hand"
                    elif kind == "target":
                        if not args.allow_live:
                            raise PermissionError("bridge was not started with --allow-live")
                        if latched:
                            raise RuntimeError("bridge is latched")
                        if expired:
                            raise TimeoutError("target command expired before receipt")
                        if len(latest) != 4:
                            raise ValueError("target packet payload missing")
                        tcp = np.frombuffer(latest[2], dtype=np.float64).reshape(4, 4).copy()
                        hand = np.frombuffer(latest[3], dtype=np.float64).reshape(16).copy()
                        if not np.all(np.isfinite(tcp)) or not np.all(np.isfinite(hand)):
                            raise ValueError("target contains non-finite values")
                        hardware.target(tcp, hand)
                        target_active = True
                        target_scope = "robot"
                        last_valid_command_ns = time.monotonic_ns()
                        status = "commanding"
                except Exception as exc:
                    if kind in {"hand_target", "hand_hold"}:
                        hardware.hold_hand()
                    else:
                        hardware.hold()
                    target_active = False
                    target_scope = None
                    latched = True
                    status = f"command_fault:{type(exc).__name__}:{exc}"

            if (
                args.watchdog_ms > 0.0
                and target_active
                and (time.monotonic_ns() - last_valid_command_ns) / 1e6 > args.watchdog_ms
            ):
                if target_scope == "hand":
                    hardware.hold_hand()
                else:
                    hardware.hold()
                target_active = False
                target_scope = None
                latched = True
                status = "watchdog_timeout"

            state, tcp, connected, error, state_ns = hardware.feedback()
            if error:
                latched = True
                status = "controller_error"
            header = _state_header(
                state_ns=state_ns,
                connected=connected,
                error=error,
                latched=latched,
                allow_live=args.allow_live,
                status=status,
            )
            state_pub.send_multipart(
                [
                    b"state",
                    json.dumps(header).encode("utf-8"),
                    np.asarray(state, dtype=np.float64).tobytes(order="C"),
                    np.asarray(tcp, dtype=np.float64).tobytes(order="C"),
                ]
            )
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        hardware.close()
        state_pub.close(0)
        command_pull.close(0)


if __name__ == "__main__":
    main()
