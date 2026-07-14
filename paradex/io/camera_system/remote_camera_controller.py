"""Main-PC controller for the capture-PC ZMQ camera agents."""

from __future__ import annotations

import threading
import time
import zmq
from concurrent.futures import ThreadPoolExecutor
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

    COMMAND_WAIT_SECONDS = 35.0

    def __init__(self, name, pc_list=None):
        self.name = "{}_{}".format(name, datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.pc_list = get_pc_list() if pc_list is None else pc_list

        self.ping_port = 5480
        self.command_port = 5482
        self.connection_port = 5483

        self.exit_event = threading.Event()
        self.start_event = threading.Event()
        self.stop_event = threading.Event()
        self.sending_event = threading.Event()
        self.error_event = threading.Event()
        self.ready_event = threading.Event()
        self._request_lock = threading.Lock()
        self._initialization_error = None
        self._command_error = None
        self._last_response: Dict[str, dict] = {}
        self._last_heartbeat_failures = {}

        self.run_thread = threading.Thread(target=self.run, daemon=True)
        self.run_thread.start()

    def initialize(self):
        self.ctx = zmq.Context()
        self.command_sockets = {}
        failed_pcs = []

        for pc in self.pc_list:
            if not self.check_server_alive(pc):
                failed_pcs.append(pc)
                continue

            socket = self.ctx.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 30000)
            socket.setsockopt(zmq.SNDTIMEO, 10000)
            socket.connect("tcp://{}:{}".format(get_pc_ip(pc), self.command_port))
            self.command_sockets[pc] = socket
            print("{}: Command socket connected".format(pc))

        if failed_pcs:
            raise ConnectionError(
                "다음 PC들이 응답하지 않습니다: {}\n"
                "각 PC에서 'python src/camera/server_daemon.py --backend aravis-gstreamer'를 실행하세요.".format(
                    failed_pcs
                )
            )

        responses = self.register()
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

    def send_command(self, cmd):
        """Send one command to every capture PC in parallel."""

        command = dict(cmd)
        command["controller_name"] = self.name

        def send_one(pc, socket):
            try:
                socket.send_json(command)
                return pc, socket.recv_json()
            except zmq.ZMQError as exc:
                return pc, {"status": "error", "msg": "no response: {}".format(exc)}

        if not self.command_sockets:
            return {}
        with ThreadPoolExecutor(max_workers=len(self.command_sockets)) as executor:
            futures = [executor.submit(send_one, pc, socket) for pc, socket in self.command_sockets.items()]
            return {pc: response for pc, response in (future.result() for future in futures)}

    def register(self):
        return self.send_command({"action": "register"})

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
        if not self.ready_event.wait(self.COMMAND_WAIT_SECONDS):
            raise RemoteCameraControllerError("Timed out initializing remote camera controller")
        if self._initialization_error is not None:
            raise RemoteCameraControllerError(
                "Remote camera controller initialization failed: {}".format(self._initialization_error)
            )

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
        return self._request(self.start_event)

    def stop(self):
        return self._request(self.stop_event)

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
            self._last_heartbeat_failures = failures
        else:
            self._last_heartbeat_failures = {}
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
            for socket in getattr(self, "command_sockets", {}).values():
                socket.close()
            if hasattr(self, "ctx"):
                self.ctx.term()
            return

        try:
            while not self.exit_event.is_set():
                action = "heartbeat"
                command = {"action": action}
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
                elif self.stop_event.is_set():
                    action = "stop"
                    command = {"action": action}
                    self.stop_event.clear()

                response = self.send_command(command)
                if action in ("start", "stop"):
                    self._complete_command(action, response)
                else:
                    failures = self._failed_responses(response)
                    if failures:
                        self.error_event.set()
                        if failures != self._last_heartbeat_failures:
                            for pc, message in failures.items():
                                print("{}: {}".format(pc, message))
                        self._last_heartbeat_failures = failures
                    else:
                        self._last_heartbeat_failures = {}
                time.sleep(0.1)
        finally:
            try:
                self.send_command({"action": "end"})
            except Exception:
                pass
            for socket in self.command_sockets.values():
                socket.close()
            self.ctx.term()

    def is_error(self):
        return self.error_event.is_set()
