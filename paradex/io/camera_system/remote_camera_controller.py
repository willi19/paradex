"""Main-PC controller for the capture-PC ZMQ camera agents."""

from __future__ import annotations

import queue
import threading
import time
import zmq
from datetime import datetime
from typing import Dict

from paradex.utils.system import get_pc_ip, get_pc_list


class RemoteCameraControllerError(RuntimeError):
    """A capture PC did not acknowledge a session-critical command."""


class remote_camera_controller:
    """Keep the existing ZMQ contract while making START a real READY barrier.

    The server replies to ``start`` only after every local Aravis/GStreamer
    pipeline is ready.  This class surfaces a failed reply to CaptureSession,
    so its UTG900E is never enabled for a partial camera group.
    """

    COMMAND_RESPONSE_SECONDS = 30.0
    COMMAND_WAIT_SECONDS = 35.0
    INITIALIZATION_WAIT_SECONDS = 60.0
    REGISTER_WAIT_SECONDS = 5.0
    REGISTER_ATTEMPTS = 3
    HEARTBEAT_INTERVAL_SECONDS = 0.5
    HEARTBEAT_RESPONSE_SECONDS = 2.0

    def __init__(self, name, pc_list=None):
        self.name = "{}_{}".format(name, datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.pc_list = get_pc_list() if pc_list is None else pc_list

        self.ping_port = 5480
        self.command_port = 5482
        self.connection_port = 5483

        self.exit_event = threading.Event()
        self.start_event = threading.Event()
        self.prepare_event = threading.Event()
        self.stop_event = threading.Event()
        self.snapshot_event = threading.Event()
        self.validate_event = threading.Event()
        self.abort_event = threading.Event()
        self.sending_event = threading.Event()
        self.error_event = threading.Event()
        self.ready_event = threading.Event()
        self._request_lock = threading.Lock()
        self._send_command_lock = threading.Lock()
        self._heartbeat_lock = threading.Lock()
        self._initialization_error = None
        self._command_error = None
        self._last_response: Dict[str, dict] = {}
        self._last_heartbeat_failures = {}

        # Each capture PC owns an independent command queue and worker.  A slow
        # response from one PC must not prevent the other PCs from receiving
        # their keepalive before the daemon's dead-man timeout expires.
        self.command_queues = {}
        self.command_sockets = {}
        self.worker_threads = {}
        self.worker_stop = threading.Event()
        self._worker_ready = {}
        self._worker_errors = {}

        self.run_thread = threading.Thread(target=self.run, daemon=True)
        self.run_thread.start()

    def initialize(self):
        self.ctx = zmq.Context()
        failed_pcs = []

        for pc in self.pc_list:
            if not self.check_server_alive(pc):
                failed_pcs.append(pc)
                continue

            self.command_queues[pc] = queue.Queue()
            self._worker_ready[pc] = threading.Event()
            thread = threading.Thread(
                target=self._command_worker,
                args=(pc,),
                name="camera-command-{}".format(pc),
                daemon=True,
            )
            self.worker_threads[pc] = thread
            thread.start()

        if failed_pcs:
            raise ConnectionError(
                "다음 PC들이 응답하지 않습니다: {}\n"
                "각 PC에서 'python src/camera/server_daemon.py --backend aravis-gstreamer'를 실행하세요.".format(
                    failed_pcs
                )
            )

        for pc, ready in self._worker_ready.items():
            if not ready.wait(self.REGISTER_WAIT_SECONDS):
                self._worker_errors[pc] = "command worker did not initialize"
        if self._worker_errors:
            raise ConnectionError(
                "Camera command workers failed: {}".format(self._worker_errors)
            )

        responses = {}
        for attempt in range(1, self.REGISTER_ATTEMPTS + 1):
            responses = self.register()
            failures = self._failed_responses(responses)
            if not failures:
                return
            retryable = all(
                message.startswith(("no response", "send failed", "receive failed"))
                for message in failures.values()
            )
            if not retryable or attempt == self.REGISTER_ATTEMPTS:
                break
            print(
                "Camera registration attempt {}/{} failed; retrying...".format(
                    attempt, self.REGISTER_ATTEMPTS
                )
            )
            time.sleep(0.5)
        self._raise_for_failed_response("register", responses)

    def check_server_alive(self, pc):
        """Ping port로 서버 확인."""

        socket = self.ctx.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 5000)
        socket.setsockopt(zmq.SNDTIMEO, 5000)
        try:
            socket.connect("tcp://{}:{}".format(get_pc_ip(pc), self.ping_port))
            socket.send_string("ping")
            return socket.recv_string() == "pong"
        except zmq.ZMQError:
            return False
        finally:
            socket.close()

    def _create_command_socket(self, pc):
        """Create a REQ socket in the worker thread that exclusively owns it."""

        socket = self.ctx.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        # Permit a worker to recover after a timed-out request instead of
        # remaining stuck in REQ's send/receive state machine.
        socket.setsockopt(zmq.REQ_RELAXED, 1)
        socket.setsockopt(zmq.REQ_CORRELATE, 1)
        socket.connect("tcp://{}:{}".format(get_pc_ip(pc), self.command_port))
        return socket

    @staticmethod
    def _socket_error(phase, exc):
        return {
            "status": "error",
            "msg": "{} failed: {}".format(phase, exc),
        }

    def _send_one(self, socket, command, timeout_seconds):
        """Send one request from its owning worker and return one response."""

        timeout_ms = max(1, int(float(timeout_seconds) * 1000))
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        try:
            socket.send_json(command)
        except zmq.ZMQError as exc:
            return self._socket_error("send", exc), False
        try:
            return socket.recv_json(), True
        except zmq.ZMQError as exc:
            return self._socket_error("receive", exc), False

    def _record_heartbeat(self, pc, response):
        failure = None
        if response.get("status") != "ok":
            failure = response.get("msg", "unknown camera-agent error")

        with self._heartbeat_lock:
            previous = self._last_heartbeat_failures.get(pc)
            if failure is None:
                self._last_heartbeat_failures.pop(pc, None)
                return
            self.error_event.set()
            self._last_heartbeat_failures[pc] = failure
            if failure != previous:
                print("{}: {}".format(pc, failure))

    def _command_worker(self, pc):
        """Own one PC's ZMQ socket and keep its daemon lease alive."""

        socket = None
        try:
            socket = self._create_command_socket(pc)
            self.command_sockets[pc] = socket
            print("{}: Command socket connected".format(pc))
        except Exception as exc:
            self._worker_errors[pc] = str(exc)
        finally:
            self._worker_ready[pc].set()

        if socket is None:
            return

        command_queue = self.command_queues[pc]
        registered = False
        try:
            while not self.worker_stop.is_set():
                try:
                    item = command_queue.get(
                        timeout=self.HEARTBEAT_INTERVAL_SECONDS
                    )
                except queue.Empty:
                    if not registered:
                        continue
                    heartbeat = {
                        "action": "heartbeat",
                        "controller_name": self.name,
                    }
                    response, reusable = self._send_one(
                        socket,
                        heartbeat,
                        self.HEARTBEAT_RESPONSE_SECONDS,
                    )
                    self._record_heartbeat(pc, response)
                    if not reusable:
                        socket.close()
                        socket = self._create_command_socket(pc)
                        self.command_sockets[pc] = socket
                    continue

                if item is None:
                    break
                command, timeout_seconds, response_queue = item
                response, reusable = self._send_one(
                    socket,
                    command,
                    timeout_seconds,
                )
                if command.get("action") == "register":
                    registered = response.get("status") == "ok"
                elif command.get("action") == "end" and response.get("status") == "ok":
                    registered = False
                response_queue.put(response)

                if not reusable:
                    socket.close()
                    socket = self._create_command_socket(pc)
                    self.command_sockets[pc] = socket
        except Exception as exc:
            self._worker_errors[pc] = str(exc)
            self.error_event.set()
        finally:
            self.command_sockets.pop(pc, None)
            socket.close()

    def _stop_command_workers(self):
        self.worker_stop.set()
        for command_queue in self.command_queues.values():
            command_queue.put(None)
        for thread in self.worker_threads.values():
            thread.join(timeout=self.HEARTBEAT_RESPONSE_SECONDS + 1.0)

    def send_command(self, cmd, timeout_seconds=None):
        """Broadcast through per-PC workers without coupling their heartbeats."""

        with self._send_command_lock:
            return self._send_command_locked(cmd, timeout_seconds)

    def _send_command_locked(self, cmd, timeout_seconds):
        """Queue one ordered rig-wide command and wait for its replies."""

        command = dict(cmd)
        command["controller_name"] = self.name

        if timeout_seconds is None:
            timeout_seconds = self.COMMAND_RESPONSE_SECONDS
        timeout_seconds = float(timeout_seconds)

        response_queues = {}
        for pc, command_queue in self.command_queues.items():
            response_queue = queue.Queue(maxsize=1)
            response_queues[pc] = response_queue
            command_queue.put((command, timeout_seconds, response_queue))

        responses = {}
        deadline = time.monotonic() + float(timeout_seconds)
        for pc, response_queue in response_queues.items():
            remaining = max(0.0, deadline - time.monotonic())
            try:
                responses[pc] = response_queue.get(timeout=remaining)
            except queue.Empty:
                responses[pc] = {
                    "status": "error",
                    "msg": "no response within {:.1f}s".format(timeout_seconds),
                }
        return responses

    def register(self):
        return self.send_command(
            {"action": "register"},
            timeout_seconds=self.REGISTER_WAIT_SECONDS,
        )

    @staticmethod
    def _failed_responses(responses):
        return {
            pc: response.get("msg", "unknown camera-agent error")
            for pc, response in responses.items()
            if response.get("status") != "ok"
        }

    def _raise_for_failed_response(self, action, responses):
        failures = self._failed_responses(responses)
        if failures:
            self.error_event.set()
            raise RemoteCameraControllerError("{} failed: {}".format(action, failures))

    def _wait_until_initialized(self):
        if not self.ready_event.wait(self.INITIALIZATION_WAIT_SECONDS):
            raise RemoteCameraControllerError("Timed out initializing remote camera controller")
        if self._initialization_error is not None:
            raise RemoteCameraControllerError(
                "Remote camera controller initialization failed: {}".format(self._initialization_error)
            )

    def wait_until_ready(self):
        """Block until every configured capture PC accepted registration."""

        self._wait_until_initialized()

    def _request(self, event):
        self._wait_until_initialized()
        with self._request_lock:
            self.sending_event.clear()
            self._command_error = None
            event.set()
            if not self.sending_event.wait(self.COMMAND_WAIT_SECONDS):
                raise RemoteCameraControllerError("Timed out waiting for camera-agent command response")
            if self._command_error is not None:
                raise RemoteCameraControllerError(self._command_error)
            return dict(self._last_response)

    def start(self, mode, syncMode, save_path=None, fps=30):
        self.mode = mode
        self.syncMode = syncMode
        self.save_path = save_path
        self.fps = fps
        # Slow camera programming/build happens on every PC before any
        # aravissrc enters PLAYING.  ACTIVATE is then a short global barrier;
        # CaptureSession enables UTG only after it completes.
        try:
            self._request(self.prepare_event)
            return self._request(self.start_event)
        except Exception:
            try:
                self._request(self.abort_event)
            except Exception:
                pass
            raise

    def stop(self):
        return self._request(self.stop_event)

    def snapshot(self, save_path):
        self.snapshot_save_path = save_path
        return self._request(self.snapshot_event)

    def validate(self, timeout=5.0):
        self.validate_timeout = timeout
        return self._request(self.validate_event)

    def end(self):
        self.exit_event.set()
        self.run_thread.join(timeout=self.COMMAND_WAIT_SECONDS)

    def reload_cameras(self):
        self._wait_until_initialized()
        response = self.send_command({"action": "reload"})
        self._raise_for_failed_response("reload", response)
        return response

    def _complete_command(self, action, response):
        self._last_response = response
        failures = self._failed_responses(response)
        if failures:
            self.error_event.set()
            self._command_error = "{} failed: {}".format(action, failures)
        self.sending_event.set()

    def run(self):
        initialized = False
        try:
            self.initialize()
            initialized = True
        except Exception as exc:
            self._initialization_error = exc
        finally:
            self.ready_event.set()

        if not initialized:
            self._stop_command_workers()
            if hasattr(self, "ctx"):
                self.ctx.term()
            return

        try:
            while not self.exit_event.is_set():
                action = None
                command = None
                if self.start_event.is_set():
                    action = "start"
                    command = {
                        "action": action,
                        "mode": self.mode,
                        "syncMode": self.syncMode,
                        "save_path": self.save_path,
                        "fps": self.fps,
                    }
                    self.start_event.clear()
                elif self.prepare_event.is_set():
                    action = "prepare"
                    command = {
                        "action": action,
                        "mode": self.mode,
                        "syncMode": self.syncMode,
                        "save_path": self.save_path,
                        "fps": self.fps,
                    }
                    self.prepare_event.clear()
                elif self.stop_event.is_set():
                    action = "stop"
                    command = {"action": action}
                    self.stop_event.clear()
                elif self.snapshot_event.is_set():
                    action = "snapshot"
                    command = {
                        "action": action,
                        "save_path": self.snapshot_save_path,
                    }
                    self.snapshot_event.clear()
                elif self.validate_event.is_set():
                    action = "validate"
                    command = {"action": action, "timeout": self.validate_timeout}
                    self.validate_event.clear()
                elif self.abort_event.is_set():
                    action = "abort"
                    command = {"action": action}
                    self.abort_event.clear()

                if command is not None:
                    response = self.send_command(command)
                    self._complete_command(action, response)
                time.sleep(0.05)
        finally:
            try:
                self.send_command({"action": "end"})
            except Exception:
                pass
            self._stop_command_workers()
            self.ctx.term()

    def is_error(self):
        return self.error_event.is_set()
