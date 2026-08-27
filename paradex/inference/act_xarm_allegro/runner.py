"""Mode implementations for contract, replay, shadow, and live execution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from pathlib import Path

import numpy as np

from paradex.inference.act_xarm_allegro.camera import CameraStreamError, SynchronizedCameraStream
from paradex.inference.act_xarm_allegro.controls import DeadmanState, KeyboardDeadman
from paradex.inference.act_xarm_allegro.core import (
    ObservationPacket,
    RunnerConfig,
    SafetyFilter,
    SafetyVerdict,
    decode_action,
)
from paradex.inference.act_xarm_allegro.policy import (
    LoadedPolicy,
    load_policy,
    validate_dataset_contract,
)
from paradex.inference.act_xarm_allegro.telemetry import RunLogger, iter_replay_packets
from paradex.inference.act_xarm_allegro.transport import HardwareBridgeClient, RobotFeedback


@dataclass(frozen=True)
class QueuedAction:
    raw: np.ndarray
    chunk: int
    step: int


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
            # A delayed, but synchronized, observation remains useful for this
            # direct runner. Do not turn an observation-age warning into a
            # command gate.
            max_frame_age_ms=float("inf"),
            manage_capture_session=manage_capture_session,
        )
        self.keyboard: KeyboardDeadman | None = None
        self.deadman = DeadmanState()
        self.queue: deque[QueuedAction] = deque()
        self._queue_lock = threading.Lock()
        self._plan_event = threading.Event()
        self._planner_stop = threading.Event()
        self._planner_thread: threading.Thread | None = None
        self._planner_enabled = config.mode == "shadow"
        self._planning = False
        self._plan_generation = 0
        self._planner_error: Exception | None = None
        self._latest_feedback: RobotFeedback | None = None
        self._refill_threshold = min(4, max(0, config.action_steps - 1))
        self._command_sequence = 0
        self._queue_starved = False
        self.previous_tcp: np.ndarray | None = None
        self.previous_hand: np.ndarray | None = None
        self.faults = 0
        self.chunks = 0
        self.chunks_in_enable = 0
        self.last_enable_generation = -1
        self.was_held = False
        self.was_aborted = False
        self._last_reported_fault: str | None = None
        self._checks_ready = False
        self._last_reported_support_warning: str | None = None

    def _packet(self, feedback: RobotFeedback) -> ObservationPacket:
        pair = self.cameras.get_pair(timeout_seconds=max(0.2, 3.0 / self.config.control_hz))
        return ObservationPacket(
            images=pair.images,
            state=feedback.state.copy(),
            frame_ids=pair.frame_ids,
            captured_monotonic_ns=pair.received_monotonic_ns,
            state_monotonic_ns=feedback.state_monotonic_ns,
            jpeg_bytes=pair.jpeg_bytes,
            raw_frame_ids=pair.raw_frame_ids,
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
        with getattr(self, "_queue_lock", threading.Lock()):
            self.queue.clear()
        self.faults += 1
        latched = latch_immediately or self.faults >= self.safety.config.max_consecutive_faults
        self.logger.event("safety_reject", reason=reason, consecutive_faults=self.faults, latched=latched)
        if reason != getattr(self, "_last_reported_fault", None):
            print(f"[act] SAFETY HOLD: {reason}")
            self._last_reported_fault = reason
        if self.config.mode == "live" and latched:
            self.deadman.press("esc")
            self.bridge.send_abort()

    def _clear_actions(self) -> None:
        with self._queue_lock:
            self.queue.clear()
            self._plan_generation += 1

    def _request_plan_if_needed(self) -> None:
        with self._queue_lock:
            chunk_limit_reached = (
                self.max_chunks is not None and self.chunks >= self.max_chunks
            )
            enable_limit_reached = (
                self.config.max_chunks_per_enable > 0
                and self.chunks_in_enable >= self.config.max_chunks_per_enable
            )
            should_plan = (
                self._planner_enabled
                and not self._planning
                and len(self.queue) <= self._refill_threshold
                and not chunk_limit_reached
                and not enable_limit_reached
            )
        if should_plan:
            self._plan_event.set()

    def _planner_loop(self) -> None:
        """Acquire observations and prepare chunks outside the 30 Hz publisher."""

        period = 1.0 / self.config.control_hz
        # ZeroMQ sockets are thread-affine. The publisher keeps self.bridge's
        # command socket; this planner owns a separate feedback socket created
        # and closed in this thread.
        feedback_bridge = HardwareBridgeClient(
            self.config.state_endpoint,
            self.config.command_endpoint,
        )
        try:
            while not self._planner_stop.is_set():
                if not self._plan_event.wait(timeout=0.1):
                    continue
                self._plan_event.clear()
                with self._queue_lock:
                    if not self._planner_enabled or self._planning:
                        continue
                    if len(self.queue) > self._refill_threshold:
                        continue
                    if self.max_chunks is not None and self.chunks >= self.max_chunks:
                        continue
                    if (
                        self.config.max_chunks_per_enable > 0
                        and self.chunks_in_enable >= self.config.max_chunks_per_enable
                    ):
                        continue
                    generation = self._plan_generation
                    self._planning = True
                    planning_started = time.monotonic()

                try:
                    feedback = feedback_bridge.receive_feedback(
                        timeout_seconds=max(0.1, 2.0 * period)
                    )
                    self._latest_feedback = feedback
                    packet = self._packet(feedback)
                    actions, inference_ms = self.policy.infer(
                        dict(packet.images), packet.state, self.config.action_steps
                    )
                except Exception as exc:
                    self.logger.event(
                        "planner_error",
                        error=f"{type(exc).__name__}:{exc}",
                    )
                    if self.config.mode == "shadow":
                        self._planner_error = exc
                        self._planner_stop.set()
                    else:
                        time.sleep(period)
                else:
                    accepted = False
                    with self._queue_lock:
                        if self._planner_enabled and generation == self._plan_generation:
                            chunk = self.chunks
                            queue_before = len(self.queue)
                            self.queue.extend(
                                QueuedAction(np.asarray(raw).copy(), chunk, step)
                                for step, raw in enumerate(actions)
                            )
                            queue_after = len(self.queue)
                            self.chunks += 1
                            self.chunks_in_enable += 1
                            accepted = True
                    if accepted:
                        self.logger.event(
                            "planner_chunk_ready",
                            chunk=chunk,
                            planning_ms=(time.monotonic() - planning_started) * 1000.0,
                            queue_before=queue_before,
                            queue_after=queue_after,
                        )
                        # The publisher can consume the new chunk while the planner
                        # writes JPEG/NPZ artifacts.
                        try:
                            self.logger.inference_boundary(packet, actions, inference_ms)
                        except Exception as exc:
                            self.logger.event(
                                "telemetry_error",
                                error=f"{type(exc).__name__}:{exc}",
                            )
                finally:
                    with self._queue_lock:
                        self._planning = False
                self._request_plan_if_needed()
        finally:
            feedback_bridge.close()

    def _start_planner(self) -> None:
        self._planner_stop.clear()
        self._planner_thread = threading.Thread(
            target=self._planner_loop,
            name="act-planner",
            daemon=True,
        )
        self._planner_thread.start()
        self._request_plan_if_needed()

    def _stop_planner(self) -> None:
        self._planner_stop.set()
        self._plan_event.set()
        if self._planner_thread is not None:
            self._planner_thread.join(timeout=1.0)
            self._planner_thread = None

    def _preposition_allegro(self, feedback: RobotFeedback) -> RobotFeedback:
        """Issue the requested hand start target without a convergence gate."""

        target = self.safety.config.allegro_start_target
        if target is None:
            return feedback
        target = np.asarray(target, dtype=np.float64).reshape(16)

        print(
            "[act] prepositioning all Allegro joints to dataset q50: "
            + np.array2string(target, precision=3, separator=", ")
        )
        self.logger.event(
            "allegro_preposition_start",
            measured=feedback.state[6:],
            target=target,
        )
        self.bridge.send_rearm()
        for _ in range(3):
            self.bridge.send_hand_target(target, ttl_ms=200.0)
            time.sleep(min(0.02, 0.5 / self.config.control_hz))
        # Do not wait for feedback convergence: it was an unnecessary live
        # startup gate and the ACT action stream begins as soon as R is pressed.
        self.bridge.send_hand_hold()
        final_feedback = feedback
        max_error = float(np.max(np.abs(feedback.state[6:] - target)))
        self.logger.event(
            "allegro_preposition_complete",
            measured=final_feedback.state[6:],
            target=target,
            max_error_rad=max_error,
        )
        print(f"[act] Allegro start target issued: measured max error={max_error:.3f} rad")
        return final_feedback

    def run(self) -> int:
        period = 1.0 / self.config.control_hz
        if self.config.mode == "live":
            self.deadman.press("esc")
            # The initial local latch must not send an abort to a healthy bridge.
            self.was_aborted = True
            self.keyboard = KeyboardDeadman(self.deadman)
            self.keyboard.start()
            print("[act] LIVE latched: press R after checks pass to run; Esc aborts")
        try:
            self.cameras.start()
            initial = self.bridge.receive_feedback(timeout_seconds=3.0)
            if self.config.mode == "live" and not initial.allow_live:
                raise RuntimeError("ROS bridge was not started with --allow-live")
            if self.config.mode == "live" and self.config.preposition_allegro:
                initial = self._preposition_allegro(initial)
            self.previous_tcp = initial.tcp_transform.copy()
            self.previous_hand = initial.state[6:].copy()
            self._latest_feedback = initial
            self._start_planner()
            # Duration describes usable rollout time, not model/camera startup time.
            started = time.monotonic()
            next_tick = started
            while True:
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(next_tick - now)
                    now = time.monotonic()
                lateness = now - next_tick
                if lateness > period:
                    self.logger.event("publisher_overrun", elapsed_ms=lateness * 1000.0)
                    next_tick = now
                scheduled_ns = int(next_tick * 1e9)
                next_tick += period

                if self.config.duration_seconds is not None and now - started >= self.config.duration_seconds:
                    break
                with self._queue_lock:
                    queue_empty = not self.queue
                    planning = self._planning
                    chunk_limit_reached = (
                        self.max_chunks is not None and self.chunks >= self.max_chunks
                    )
                if chunk_limit_reached and queue_empty and not planning:
                    break

                if self._planner_error is not None:
                    raise RuntimeError("ACT planner failed") from self._planner_error

                snapshot = self.deadman.snapshot()
                if self.config.mode == "live" and not self._checks_ready:
                    self._checks_ready = True
                    print("[act] startup ready: press R to arm and run; Esc aborts")
                if self.config.mode == "live" and self.deadman.consume_rearm(True):
                    self.bridge.send_rearm()
                    self.faults = 0
                    self._clear_actions()
                    with self._queue_lock:
                        self._planner_enabled = True
                        self.chunks_in_enable = 0
                    feedback = self._latest_feedback or initial
                    self.previous_tcp = feedback.tcp_transform.copy()
                    self.previous_hand = feedback.state[6:].copy()
                    self._last_reported_fault = None
                    self.logger.event("rearmed")
                    print("[act] ARMED: executing continuously; Esc aborts and holds")
                    self._request_plan_if_needed()
                    continue

                enabled = self.config.mode == "shadow" or not snapshot.aborted
                if snapshot.enable_generation != self.last_enable_generation:
                    self.last_enable_generation = snapshot.enable_generation
                    self.chunks_in_enable = 0
                if snapshot.aborted and not self.was_aborted and self.config.mode == "live":
                    with self._queue_lock:
                        self._planner_enabled = False
                    self._clear_actions()
                    self.bridge.send_abort()
                    self.logger.event("operator_abort")
                self.was_held = snapshot.held
                self.was_aborted = snapshot.aborted

                if enabled:
                    self._request_plan_if_needed()
                    with self._queue_lock:
                        queued = self.queue.popleft() if self.queue else None
                        queue_remaining = len(self.queue)
                    if queued is not None:
                        if self._queue_starved:
                            self._queue_starved = False
                            self.logger.event("queue_recovered")
                        try:
                            decoded = decode_action(queued.raw)
                            verdict = (
                                self.safety.validate_action(
                                    decoded, self.previous_tcp, self.previous_hand
                                )
                                if self.config.enforce_safety_gates
                                else SafetyVerdict(True, "direct", decoded)
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
                                self._command_sequence += 1
                                self.logger.event(
                                    "action",
                                    sequence=self._command_sequence,
                                    chunk=queued.chunk,
                                    step=queued.step,
                                    scheduled_monotonic_ns=scheduled_ns,
                                    sent_monotonic_ns=time.monotonic_ns(),
                                    queue_remaining=queue_remaining,
                                    published=self.config.mode == "live",
                                )
                            else:
                                self._fault(verdict.reason)
                    elif not self._queue_starved:
                        self._queue_starved = True
                        self.logger.event("queue_starved")
                    self._request_plan_if_needed()
            return self.chunks
        finally:
            self._stop_planner()
            if self.config.mode == "live":
                try:
                    self.bridge.send_hold()
                except Exception:
                    pass
            if self.keyboard is not None:
                self.keyboard.close()
            self.cameras.close()
            self.bridge.close()
