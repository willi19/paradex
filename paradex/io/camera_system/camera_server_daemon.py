"""ZMQ camera-agent server used on every capture PC."""

from __future__ import annotations

import threading
import time
import traceback
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional
from urllib.parse import unquote, urlparse

import zmq

from paradex.io.camera_system.camera_loader import CameraLoader


class camera_server_daemon:
    """Serve the legacy camera ZMQ protocol with a selectable local backend.

    ``aravis-gstreamer`` is the default backend.  It owns only local camera
    discovery/configuration/pipelines; it intentionally has no UTG900E code.
    ``pyspin`` remains available as an explicit rollback backend.
    """

    def __init__(
        self,
        backend: str = "aravis-gstreamer",
        loader=None,
        loader_factory: Optional[Callable[[], object]] = None,
        start_threads: bool = True,
        preview_port: int = 5484,
    ):
        self.backend = backend
        self._loader_factory = loader_factory or self._make_loader_factory(backend)
        self.camera_loader = loader if loader is not None else self._loader_factory()

        self.ping_port = 5480
        self.monitor_port = 5481
        self.command_port = 5482
        self.connection_port = 5483
        self.preview_port = int(preview_port)
        self.ctx = zmq.Context()
        self.current_controller = None
        self.state = "idle"
        self._threads = []
        self._shutdown_event = threading.Event()
        self._close_lock = threading.Lock()
        self._camera_lock = threading.RLock()
        self._preview_server = None

        if start_threads:
            self._start_threads()

    @staticmethod
    def _make_loader_factory(backend: str) -> Callable[[], object]:
        normalized = backend.lower().replace("_", "-")
        if normalized == "pyspin":
            return CameraLoader
        if normalized in ("aravis", "aravis-gstreamer"):
            from paradex.io.camera_system.aravis_gstreamer import AravisGStreamerCameraLoader

            return AravisGStreamerCameraLoader
        raise ValueError("Unknown camera backend {!r}; use pyspin or aravis-gstreamer".format(backend))

    def _start_threads(self):
        for target in (self.pingpong_thread, self.monitor_thread, self.command_thread):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)
        self._start_preview_server()

    def _start_preview_server(self):
        daemon = self

        class PreviewHandler(BaseHTTPRequestHandler):
            def _send_json(self, status, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/cameras":
                    self._send_json(
                        200,
                        {
                            "backend": daemon.backend,
                            "state": daemon.state,
                            "cameras": daemon.camera_loader.get_status_list(),
                        },
                    )
                    return
                if path.startswith("/preview/"):
                    serial = unquote(path[len("/preview/") :])
                    get_preview = getattr(daemon.camera_loader, "get_preview", None)
                    jpeg = get_preview(serial) if get_preview is not None else None
                    if jpeg is None:
                        self._send_json(404, {"error": "preview unavailable", "serial": serial})
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(jpeg)
                    return
                self._send_json(404, {"error": "not found"})

            def log_message(self, _format, *_args):
                return

        self._preview_server = ThreadingHTTPServer(("0.0.0.0", self.preview_port), PreviewHandler)
        thread = threading.Thread(
            target=self._preview_server.serve_forever,
            name="camera-preview-http",
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)
        print("[Info] Preview image API listening on port {}.".format(self.preview_port))

    def reload_cameras(self):
        with self._camera_lock:
            self.camera_loader.end()
            time.sleep(1)
            self.camera_loader = self._loader_factory()
        print("[Info] {} camera loader reloaded.".format(self.backend))

    def close(self):
        """Finalize cameras and stop all ZMQ resources; safe to call twice."""

        with self._close_lock:
            if self._shutdown_event.is_set():
                return
            print("[Info] Shutting down {} camera agent...".format(self.backend))
            self._shutdown_event.set()
            try:
                with self._camera_lock:
                    self.camera_loader.end()
            except Exception:
                print("[Error] Camera cleanup failed during shutdown:")
                traceback.print_exc()
            finally:
                self.current_controller = None
                self.state = "closed"

            if self._preview_server is not None:
                self._preview_server.shutdown()
                self._preview_server.server_close()
                self._preview_server = None

            # command_thread can be waiting on its five-second receive timeout.
            for thread in self._threads:
                thread.join(timeout=6.0)
            self.ctx.destroy(linger=0)
            print("[Info] Camera agent shutdown complete.")

    def pingpong_thread(self):
        socket = self.ctx.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind("tcp://*:{}".format(self.ping_port))
        try:
            while not self._shutdown_event.is_set():
                try:
                    socket.recv_string(flags=zmq.NOBLOCK)
                    socket.send_string("pong")
                except zmq.ZMQError:
                    time.sleep(0.1)
        finally:
            socket.close(linger=0)

    def monitor_thread(self):
        socket = self.ctx.socket(zmq.PUB)
        socket.bind("tcp://*:{}".format(self.monitor_port))
        try:
            while not self._shutdown_event.is_set():
                status = {
                    "backend": self.backend,
                    "cameras": self.camera_loader.get_status_list(),
                    "controller": self.current_controller or "None",
                }
                socket.send_json(status)
                time.sleep(0.1)
        finally:
            socket.close(linger=0)

    def _locked_response(self, controller_name):
        if controller_name != self.current_controller and self.current_controller is not None:
            print("[Warning] {} tried to access, but locked by {}".format(controller_name, self.current_controller))
            return {"status": "error", "msg": "locked by {}".format(self.current_controller)}
        return None

    def execute_command(self, cmd):
        action = cmd.get("action")
        controller_name = cmd.get("controller_name")
        locked = self._locked_response(controller_name)
        if locked is not None:
            return locked

        if action == "register":
            self.current_controller = controller_name
            return {"status": "ok", "msg": "registered", "backend": self.backend}
        if self.current_controller is None:
            return {"status": "error", "msg": "no active controller"}

        try:
            if action == "prepare":
                self.state = "preparing"
                prepare = getattr(self.camera_loader, "prepare", None)
                if prepare is not None:
                    with self._camera_lock:
                        prepare(
                            cmd.get("mode"),
                            cmd.get("syncMode"),
                            cmd.get("save_path"),
                            cmd.get("fps", 30),
                        )
                self.state = "prepared"
                return {"status": "ok", "msg": "prepared"}
            if action == "start":
                self.state = "starting"
                with self._camera_lock:
                    activate = getattr(self.camera_loader, "activate", None)
                    if activate is not None:
                        activate()
                    else:
                        # Preserve the legacy PySpin loader contract.
                        self.camera_loader.start(
                            cmd.get("mode"),
                            cmd.get("syncMode"),
                            cmd.get("save_path"),
                            cmd.get("fps", 30),
                        )
                self.state = "capturing"
                # This is the main-PC barrier: every local pipeline has
                # completed camera setup and state preparation at this point.
                return {"status": "ok", "msg": "ready"}
            if action == "stop":
                self.state = "stopping"
                with self._camera_lock:
                    self.camera_loader.stop()
                self.state = "idle"
                return {"status": "ok", "msg": "stopped"}
            if action == "validate":
                with self._camera_lock:
                    self.camera_loader.wait_for_first_frames(cmd.get("timeout"))
                return {"status": "ok", "msg": "frames received"}
            if action == "abort":
                with self._camera_lock:
                    abort = getattr(self.camera_loader, "abort", None)
                    if abort is not None:
                        abort()
                    else:
                        self.camera_loader.stop()
                self.state = "idle"
                return {"status": "ok", "msg": "aborted"}
            if action == "end":
                with self._camera_lock:
                    self.camera_loader.stop()
                self.current_controller = None
                self.state = "idle"
                return {"status": "ok", "msg": "ended"}
            if action == "heartbeat":
                errors = self.camera_loader.get_all_errors()
                if not errors:
                    return {"status": "ok", "msg": "heartbeat received"}
                return {"status": "error", "msg": "camera errors detected: {}".format(errors)}
            if action == "reload":
                self.reload_cameras()
                return {"status": "ok", "msg": "cameras reloaded"}
            return {"status": "error", "msg": "unknown action"}
        except Exception as exc:
            self.state = "error"
            print(
                "[Error] Camera command {!r} failed: {}: {}".format(
                    action, type(exc).__name__, exc
                ),
                flush=True,
            )
            traceback.print_exc()
            return {
                "status": "error",
                "msg": "{}: {}".format(type(exc).__name__, exc),
                "traceback": traceback.format_exc(),
            }

    def command_thread(self):
        socket = self.ctx.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 5000)
        socket.bind("tcp://*:{}".format(self.command_port))
        try:
            while not self._shutdown_event.is_set():
                try:
                    command = socket.recv_json()
                    socket.send_json(self.execute_command(command))
                except zmq.Again:
                    if self._shutdown_event.is_set():
                        break
                    if self.current_controller is not None:
                        try:
                            with self._camera_lock:
                                self.camera_loader.stop()
                        except Exception:
                            traceback.print_exc()
                        self.current_controller = None
                        self.state = "idle"
                        print("[Error] Command socket timeout. Camera loader stopped and controller released.")
                except Exception as exc:
                    if self._shutdown_event.is_set():
                        break
                    traceback.print_exc()
                    try:
                        with self._camera_lock:
                            self.camera_loader.stop()
                    except Exception:
                        traceback.print_exc()
                    self.current_controller = None
                    self.state = "error"
                    try:
                        socket.send_json({"status": "error", "msg": "{}: {}".format(type(exc).__name__, exc)})
                    except zmq.ZMQError:
                        pass
        finally:
            socket.close(linger=0)
