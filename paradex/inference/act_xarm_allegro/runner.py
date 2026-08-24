"""Mode implementations for contract, replay, shadow, and live execution."""

from __future__ import annotations

from collections import deque
import time
from pathlib import Path

import numpy as np

from paradex.inference.act_xarm_allegro.camera import CameraStreamError, SynchronizedCameraStream
from paradex.inference.act_xarm_allegro.controls import DeadmanState, KeyboardDeadman
from paradex.inference.act_xarm_allegro.core import (
    ObservationPacket,
    RunnerConfig,
    SafetyFilter,
    decode_action,
)
from paradex.inference.act_xarm_allegro.policy import (
    LoadedPolicy,
    load_policy,
    validate_dataset_contract,
)
from paradex.inference.act_xarm_allegro.telemetry import RunLogger, iter_replay_packets
from paradex.inference.act_xarm_allegro.transport import HardwareBridgeClient, RobotFeedback


def _synthetic_observation(policy: LoadedPolicy) -> tuple[dict[str, np.ndarray], np.ndarray]:
    images = {
        key: np.zeros((480, 640, 3), dtype=np.uint8)
        for key in policy.contract.image_keys
    }
    return images, np.zeros(22, dtype=np.float32)


def run_contract(config: RunnerConfig) -> dict[str, object]:
    policy = load_policy(config)
    dataset_contract = validate_dataset_contract(config.dataset_repo_id)
    images, state = _synthetic_observation(policy)
    actions, inference_ms = policy.infer(images, state, config.action_steps)
    return {
        "mode": "contract",
        "contract": policy.contract,
        "dataset_contract": dataset_contract,
        "action_shape": actions.shape,
        "finite": bool(np.all(np.isfinite(actions))),
        "inference_ms": inference_ms,
        "snapshot_path": policy.snapshot_path,
    }


def run_replay(config: RunnerConfig, replay_dir: Path, logger: RunLogger, *, max_chunks: int | None = None) -> int:
    policy = load_policy(config)
    completed = 0
    for packet in iter_replay_packets(replay_dir):
        actions, inference_ms = policy.infer(dict(packet.images), packet.state, config.action_steps)
        logger.inference_boundary(packet, actions, inference_ms)
        logger.event("replay_result", finite=bool(np.all(np.isfinite(actions))))
        completed += 1
        if max_chunks is not None and completed >= max_chunks:
            break
    if completed == 0:
        raise ValueError(f"No replay chunks found in {replay_dir}")
    return completed


class HardwareRunner:
    def __init__(
        self,
        config: RunnerConfig,
        safety: SafetyFilter,
        logger: RunLogger,
        *,
        max_chunks: int | None = None,
        manage_capture_session: bool = True,
    ) -> None:
        self.config = config
        self.safety = safety
        self.logger = logger
        self.max_chunks = max_chunks
        self.manage_capture_session = manage_capture_session
        self.policy = load_policy(config)
        self.bridge = HardwareBridgeClient(config.state_endpoint, config.command_endpoint)
        self.cameras = SynchronizedCameraStream(
            config.camera_bindings,
            fps=config.control_hz,
            max_frame_age_ms=safety.config.max_observation_age_ms,
            manage_capture_session=manage_capture_session,
        )
        self.keyboard: KeyboardDeadman | None = None
        self.deadman = DeadmanState()
        self.queue: deque[np.ndarray] = deque()
        self.previous_tcp: np.ndarray | None = None
        self.previous_hand: np.ndarray | None = None
        self.faults = 0
        self.chunks = 0
        self.chunks_in_enable = 0
        self.last_enable_generation = -1
        self.was_held = False
        self.was_aborted = False

    def _packet(self, feedback: RobotFeedback) -> ObservationPacket:
        pair = self.cameras.get_pair(timeout_seconds=max(0.2, 3.0 / self.config.control_hz))
        return ObservationPacket(
            images=pair.images,
            state=feedback.state.copy(),
            frame_ids=pair.frame_ids,
            captured_monotonic_ns=pair.received_monotonic_ns,
            state_monotonic_ns=feedback.state_monotonic_ns,
            jpeg_bytes=pair.jpeg_bytes,
        )

    def _checks(
        self,
        packet: ObservationPacket,
        feedback: RobotFeedback,
        *,
        allow_latched: bool = False,
    ) -> tuple[bool, str]:
        if not feedback.connected or feedback.error or (feedback.latched and not allow_latched):
            return False, f"bridge:{feedback.status}"
        freshness = self.safety.validate_freshness(
            packet.captured_monotonic_ns,
            packet.state_monotonic_ns,
            time.monotonic_ns(),
        )
        if not freshness.accepted:
            return False, freshness.reason
        start = self.safety.validate_start(packet.state, feedback.tcp_transform)
        return start.accepted, start.reason

    def _fault(self, reason: str, *, latch_immediately: bool = False) -> None:
        self.queue.clear()
        self.faults += 1
        latched = latch_immediately or self.faults >= self.safety.config.max_consecutive_faults
        self.logger.event("safety_reject", reason=reason, consecutive_faults=self.faults, latched=latched)
        if self.config.mode == "live" and latched:
            self.deadman.press("esc")
            self.bridge.send_abort()

    def run(self) -> int:
        period = 1.0 / self.config.control_hz
        started = time.monotonic()
        if self.config.mode == "live":
            self.deadman.press("esc")
            self.keyboard = KeyboardDeadman(self.deadman)
            self.keyboard.start()
            print("[act] LIVE latched: press R after checks pass, then hold Space; Esc aborts")
        try:
            self.cameras.start()
            initial = self.bridge.receive_feedback(timeout_seconds=3.0)
            if self.config.mode == "live" and not initial.allow_live:
                raise RuntimeError("ROS bridge was not started with --allow-live")
            self.previous_tcp = initial.tcp_transform.copy()
            self.previous_hand = initial.state[6:].copy()
            while True:
                tick_started = time.monotonic()
                if self.config.duration_seconds is not None and tick_started - started >= self.config.duration_seconds:
                    break
                if self.max_chunks is not None and self.chunks >= self.max_chunks and not self.queue:
                    break
                try:
                    feedback = self.bridge.receive_feedback(timeout_seconds=max(0.1, 2.0 * period))
                    packet = self._packet(feedback)
                except (TimeoutError, CameraStreamError, RuntimeError) as exc:
                    self._fault(f"input_error:{type(exc).__name__}:{exc}", latch_immediately=True)
                    if self.config.mode == "shadow":
                        raise
                    time.sleep(period)
                    continue

                checks_passed, check_reason = self._checks(packet, feedback)
                rearm_checks_passed, _ = self._checks(packet, feedback, allow_latched=True)
                snapshot = self.deadman.snapshot()
                if self.config.mode == "live" and self.deadman.consume_rearm(rearm_checks_passed):
                    self.bridge.send_rearm()
                    self.faults = 0
                    self.queue.clear()
                    self.previous_tcp = feedback.tcp_transform.copy()
                    self.previous_hand = feedback.state[6:].copy()
                    self.logger.event("rearmed")
                    time.sleep(period)
                    continue

                enabled = self.config.mode == "shadow" or (snapshot.held and not snapshot.aborted)
                if snapshot.enable_generation != self.last_enable_generation:
                    self.last_enable_generation = snapshot.enable_generation
                    self.chunks_in_enable = 0
                if snapshot.aborted and not self.was_aborted and self.config.mode == "live":
                    self.queue.clear()
                    self.bridge.send_abort()
                    self.logger.event("operator_abort")
                elif self.was_held and not snapshot.held and self.config.mode == "live":
                    self.queue.clear()
                    self.bridge.send_hold()
                    self.previous_tcp = feedback.tcp_transform.copy()
                    self.previous_hand = feedback.state[6:].copy()
                    self.logger.event("deadman_released")
                self.was_held = snapshot.held
                self.was_aborted = snapshot.aborted

                if not checks_passed:
                    self._fault(check_reason, latch_immediately=check_reason.startswith("bridge:"))
                elif enabled:
                    if not self.queue:
                        if self.config.mode == "live" and (
                            self.config.max_chunks_per_enable > 0
                            and self.chunks_in_enable >= self.config.max_chunks_per_enable
                        ):
                            if self.config.mode == "live":
                                self.bridge.send_hold()
                        else:
                            try:
                                actions, inference_ms = self.policy.infer(
                                    dict(packet.images), packet.state, self.config.action_steps
                                )
                                self.logger.inference_boundary(packet, actions, inference_ms)
                            except Exception as exc:
                                self._fault(
                                    f"policy_error:{type(exc).__name__}:{exc}",
                                    latch_immediately=True,
                                )
                                if self.config.mode == "shadow":
                                    raise
                            else:
                                self.queue.extend(actions)
                                self.chunks += 1
                                self.chunks_in_enable += 1
                    if self.queue:
                        raw = self.queue.popleft()
                        try:
                            decoded = decode_action(raw)
                            verdict = self.safety.validate_action(
                                decoded, self.previous_tcp, self.previous_hand
                            )
                        except ValueError as exc:
                            verdict = None
                            self._fault(f"decode_error:{exc}")
                        if verdict is not None:
                            if verdict.accepted:
                                self.faults = 0
                                bounded = verdict.bounded_action
                                if self.config.mode == "live":
                                    self.bridge.send_target(
                                        bounded.tcp_transform,
                                        bounded.allegro_target,
                                        ttl_ms=max(100.0, period * 3000.0),
                                    )
                                self.previous_tcp = bounded.tcp_transform.copy()
                                self.previous_hand = bounded.allegro_target.copy()
                                self.logger.event(
                                    "action",
                                    deadman=snapshot,
                                    safety=verdict,
                                    commanded_tcp=bounded.tcp_transform,
                                    commanded_hand=bounded.allegro_target,
                                    queue_remaining=len(self.queue),
                                    published=self.config.mode == "live",
                                )
                            else:
                                self._fault(verdict.reason)
                elapsed = time.monotonic() - tick_started
                if elapsed > period:
                    self.logger.event("timing_overrun", elapsed_ms=elapsed * 1000.0)
                else:
                    time.sleep(period - elapsed)
            return self.chunks
        finally:
            if self.config.mode == "live":
                try:
                    self.bridge.send_hold()
                except Exception:
                    pass
            if self.keyboard is not None:
                self.keyboard.close()
            self.cameras.close()
            self.bridge.close()
