"""Small versioned ZMQ protocol between CUDA inference and ROS hardware."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Optional

import numpy as np
import zmq

PROTOCOL_VERSION = 1
STATE_DIM = 22


@dataclass(frozen=True)
class RobotFeedback:
    state: np.ndarray
    tcp_transform: np.ndarray
    state_monotonic_ns: int
    bridge_monotonic_ns: int
    connected: bool
    error: bool
    latched: bool
    allow_live: bool
    status: str


class HardwareBridgeClient:
    def __init__(
        self,
        state_endpoint: str,
        command_endpoint: str,
        *,
        timeout_ms: int = 2000,
    ):
        self.context = zmq.Context.instance()
        self.state_sub = self.context.socket(zmq.SUB)
        self.state_sub.setsockopt(zmq.SUBSCRIBE, b"state")
        self.state_sub.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self.state_sub.connect(state_endpoint)
        self.command_push = self.context.socket(zmq.PUSH)
        self.command_push.connect(command_endpoint)
        self._sequence = 0

    def receive_feedback(self, *, timeout_seconds: float = 2.0) -> RobotFeedback:
        deadline = time.monotonic() + float(timeout_seconds)
        latest = None
        while time.monotonic() < deadline:
            try:
                parts = self.state_sub.recv_multipart(flags=zmq.NOBLOCK)
                latest = parts
                while True:
                    parts = self.state_sub.recv_multipart(flags=zmq.NOBLOCK)
                    latest = parts
            except zmq.Again:
                if latest is not None:
                    break
                time.sleep(0.002)
        if latest is None:
            raise TimeoutError("Timed out waiting for ROS hardware bridge feedback")
        if len(latest) != 4 or latest[0] != b"state":
            raise RuntimeError("Invalid hardware bridge state packet")
        header = json.loads(latest[1].decode("utf-8"))
        if header.get("version") != PROTOCOL_VERSION:
            raise RuntimeError(f"Unsupported bridge protocol: {header.get('version')}")
        state = np.frombuffer(latest[2], dtype=np.float64).copy()
        tcp = np.frombuffer(latest[3], dtype=np.float64).reshape(4, 4).copy()
        if state.shape != (STATE_DIM,):
            raise RuntimeError(f"Bridge returned state shape {state.shape}, expected {(STATE_DIM,)}")
        return RobotFeedback(
            state=state,
            tcp_transform=tcp,
            state_monotonic_ns=int(header["state_monotonic_ns"]),
            bridge_monotonic_ns=int(header["bridge_monotonic_ns"]),
            connected=bool(header["connected"]),
            error=bool(header["error"]),
            latched=bool(header["latched"]),
            allow_live=bool(header["allow_live"]),
            status=str(header.get("status", "")),
        )

    def send_target(
        self,
        tcp_transform: np.ndarray,
        allegro_target: np.ndarray,
        *,
        ttl_ms: float = 100.0,
    ) -> None:
        self._send("target", tcp_transform, allegro_target, ttl_ms=ttl_ms)

    def send_hold(self) -> None:
        self._send("hold")

    def send_abort(self) -> None:
        self._send("abort")

    def send_rearm(self) -> None:
        self._send("rearm")

    def _send(
        self,
        kind: str,
        tcp_transform: Optional[np.ndarray] = None,
        allegro_target: Optional[np.ndarray] = None,
        *,
        ttl_ms: float = 100.0,
    ) -> None:
        self._sequence += 1
        header = {
            "version": PROTOCOL_VERSION,
            "sequence": self._sequence,
            "kind": kind,
            "issued_monotonic_ns": time.monotonic_ns(),
            "ttl_ms": float(ttl_ms),
        }
        parts = [b"command", json.dumps(header).encode("utf-8")]
        if kind == "target":
            tcp = np.asarray(tcp_transform, dtype=np.float64)
            hand = np.asarray(allegro_target, dtype=np.float64).reshape(-1)
            if tcp.shape != (4, 4) or hand.shape != (16,):
                raise ValueError("target command requires tcp (4,4) and Allegro (16,)")
            parts.extend([tcp.tobytes(order="C"), hand.tobytes(order="C")])
        self.command_push.send_multipart(parts)

    def close(self) -> None:
        self.state_sub.close(0)
        self.command_push.close(0)
