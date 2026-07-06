import os
import threading
import zmq
import time
import traceback

from paradex.io.camera_system.camera_loader import CameraLoader

class camera_server_daemon:
    def __init__(self, idle_timeout_s=None):
        self.camera_loader = CameraLoader()

        self.ping_port = 5480
        self.monitor_port = 5481
        self.command_port = 5482

        # Dead-man switch: if the controller sends no command (heartbeat) within
        # this window the daemon releases the lock and stops the cameras — the only
        # app-independent recovery when a controller crashes. Resolve:
        # arg > env PARADEX_CAMERA_IDLE_TIMEOUT_S > default 5s.
        if idle_timeout_s is None:
            idle_timeout_s = float(os.environ.get("PARADEX_CAMERA_IDLE_TIMEOUT_S", 5.0))
        self.idle_timeout_s = idle_timeout_s
        self.idle_timeout_ms = int(idle_timeout_s * 1000)

        self.ctx = zmq.Context()

        self.current_controller = None
        self.last_action = None
        self.last_action_time = None
        self.last_mode = None
        self.cameras_running = False

        self.state = "idle"

        threading.Thread(target=self.pingpong_thread, daemon=True).start()
        threading.Thread(target=self.monitor_thread, daemon=True).start()
        threading.Thread(target=self.command_thread, daemon=True).start()

    def reload_cameras(self):
        self.camera_loader.end()
        time.sleep(1)

        self.camera_loader = CameraLoader()
        print("[Info] Camera loader reloaded.")

    def shutdown(self):
        """Release every camera (DeInit + free SHM) on a clean exit.

        Call from a SIGTERM/SIGINT handler so a normal kill releases the hardware;
        SIGKILL (-9) cannot be caught (next daemon start cleans up leaked SHM)."""
        print("[Info] Shutting down: releasing cameras...")
        try:
            self.camera_loader.end()
        except Exception as e:
            print(f"[Warning] shutdown: camera_loader.end() failed: {e}")

    def pingpong_thread(self):
        self.ping_socket = self.ctx.socket(zmq.REP)
        self.ping_socket.setsockopt(zmq.LINGER, 0)
        self.ping_socket.bind(f"tcp://*:{self.ping_port}")

        while True:
            try:
                _ = self.ping_socket.recv_string(flags=zmq.NOBLOCK)
                self.ping_socket.send_string("pong")
            except zmq.ZMQError:
                time.sleep(0.1)

    def monitor_thread(self):
        """Broadcast this daemon's health on the PUB port (5481) ~10 Hz.

        Two consumers subscribe to the same stream:

        * the monitor dashboard, which reads ``cameras`` / ``controller``;
        * ``remote_camera_controller``, which reads ``summary`` / ``running`` /
          ``controller`` for its live health view (so health no longer has to
          ride the command REQ/REP reply — see the redesign in
          ``design/camera-recording-redesign.md``).

        Fire-and-forget: a subscriber that misses a message just picks up the
        next one 0.1 s later.
        """
        monitor_socket = self.ctx.socket(zmq.PUB)
        monitor_socket.bind(f"tcp://*:{self.monitor_port}")

        while True:
            status = {
                # dashboard-facing (unchanged keys, kept for back-compat)
                'cameras': self.camera_loader.get_status_list(),
                'controller': self.current_controller if self.current_controller else 'None',
                # controller-facing health telemetry
                'running': self.cameras_running,
                'summary': self.camera_loader.get_summary(),
            }
            monitor_socket.send_json(status)
            time.sleep(0.1)

    def execute_command(self, cmd):
        action = cmd.get('action')
        controller_name = cmd.get('controller_name')
        force = cmd.get('force', False)

        if action == "register":
            prev = self.current_controller
            if self.cameras_running:
                try:
                    self.camera_loader.stop()
                except Exception as e:
                    print(f"[Warning] stop on register failed: {e}")
                self.cameras_running = False
            self.current_controller = controller_name
            self.last_action = action
            self.last_action_time = time.time()
            if prev and prev != controller_name:
                print(f"[Info] controller takeover: '{prev}' -> '{controller_name}' (force={force})")
            return {"status":"ok", "msg":"registered"}

        if controller_name != self.current_controller and self.current_controller is not None:
            print(f"[Warning] {controller_name} tried to access, but locked by {self.current_controller}")
            return {"status":"error", "msg":f"locked by {self.current_controller}"}

        self.last_action = action
        self.last_action_time = time.time()

        if self.current_controller is None:
            return {"status":"error", "msg":"no active controller"}

        if action == "start":
            t0 = time.time()
            try:
                self.last_mode = cmd.get('mode')
                self.camera_loader.start(
                            cmd.get('mode'),
                            cmd.get('syncMode'),
                            cmd.get('save_path'),
                            cmd.get('fps', 30),
                            cmd.get('exposure_time'),
                            cmd.get('gain')
                        )
                dt = time.time() - t0
                # A camera can fail to arm without camera_loader.start() raising
                # (e.g. a per-camera start() timeout or BeginAcquisition error).
                # Check the error state before reporting success.
                errs = self.camera_loader.get_all_errors()
                payload = self.camera_loader.get_summary()
                payload["running"] = False
                if errs:
                    detail = {name: msg for name, (msg, tb) in errs.items()}
                    print(f"[Warning] start: {len(errs)} camera(s) failed to arm: {detail}")
                    payload.update({"status": "error", "msg": f"start: camera errors: {detail}"})
                    return payload
                self.cameras_running = cmd.get('mode') in ("video", "stream", "full")
                payload["running"] = self.cameras_running
                print(f"[Info] start completed in {dt:.2f}s mode={cmd.get('mode')} sync={cmd.get('syncMode')} fps={cmd.get('fps',30)} exposure_time={cmd.get('exposure_time')} gain={cmd.get('gain')}")
                payload.update({"status":"ok", "msg":"started"})
                return payload

            except Exception as e:
                dt = time.time() - t0
                traceback.print_exc()
                payload = self.camera_loader.get_summary()
                payload.update({"status":"error", "msg":f"start failed after {dt:.2f}s: {type(e).__name__}:{e}", "running": False})
                return payload

        if action == "stop":
            t0 = time.time()
            try:
                self.camera_loader.stop()
                self.cameras_running = False
                dt = time.time() - t0
                print(f"[Info] stop completed in {dt:.2f}s")
                payload = self.camera_loader.get_summary()
                payload.update({"status":"ok", "msg":"stopped", "running": False})
                return payload
            except Exception as e:
                dt = time.time() - t0
                traceback.print_exc()
                payload = self.camera_loader.get_summary()
                payload.update({"status":"error", "msg":f"stop failed after {dt:.2f}s: {type(e).__name__}:{e}", "running": self.cameras_running})
                return payload

        if action == "end":
            try:
                if self.cameras_running:
                    self.camera_loader.stop()
                    self.cameras_running = False
                self.current_controller = None
                self.last_mode = None
                payload = self.camera_loader.get_summary()
                payload.update({"status":"ok", "msg":"ended", "running": False})
                return payload
            except Exception:
                payload = self.camera_loader.get_summary()
                payload.update({"status":"error", "msg":"end failed", "running": self.cameras_running})
                return payload

        if action == "heartbeat":
            t0 = time.time()
            errs = self.camera_loader.get_all_errors()
            # Per-camera frame ids so the controller can detect stalls (frames that
            # stop arriving without raising an error).
            resp = self.camera_loader.get_summary()
            resp["running"] = self.cameras_running
            dt = time.time() - t0
            if dt > 0.5:
                print(f"[Warning] heartbeat get_all_errors took {dt*1000:.0f}ms (>500ms)")
            if len(errs) == 0:
                resp.update({"status": "ok", "msg": "heartbeat received"})
            else:
                resp.update({"status": "error", "msg": f"camera errors detected: {errs}"})
            return resp

        if action == "reload":
            try:
                self.reload_cameras()
                return {"status":"ok", "msg":"cameras reloaded"}
            except Exception:
                return {"status":"error", "msg":"reload failed"}
            
        return {"status":"error", "msg":"unknown action"}


    def command_thread(self):
        self.command_socket = self.ctx.socket(zmq.REP)
        self.command_socket.setsockopt(zmq.RCVTIMEO, self.idle_timeout_ms)  # dead-man timeout
        self.command_socket.bind(f"tcp://*:{self.command_port}")
        
        while True:
            try:
                cmd = self.command_socket.recv_json()
                response = self.execute_command(cmd)
                    
                self.command_socket.send_json(response)
                
            except zmq.Again:
                if self.current_controller is not None:
                    idle = (time.time() - self.last_action_time) if self.last_action_time else -1
                    released = self.current_controller
                    last_act = self.last_action
                    mode = self.last_mode
                    running = self.cameras_running
                    if running:
                        self.camera_loader.stop()
                        self.cameras_running = False
                    self.current_controller = None
                    self.last_mode = None
                    print(
                        f"[Info] Idle timeout (>{self.idle_timeout_s}s, actual={idle:.1f}s): "
                        f"released controller='{released}' last_action='{last_act}' "
                        f"mode='{mode}' cameras_were_running={running}. "
                        f"Cause: controller did not send heartbeat/end within {self.idle_timeout_s}s."
                    )

            except Exception as e:
                if self.cameras_running:
                    self.camera_loader.stop()
                    self.cameras_running = False
                released = self.current_controller
                last_act = self.last_action
                self.current_controller = None
                self.last_mode = None

                traceback.print_exc()
                self.command_socket.send_json({
                    'status': 'error',
                    'msg': f'{type(e).__name__}: {str(e)} traceback : {traceback.format_exc()}'
                })
                print(
                    f"[Error] Exception in command thread: {type(e).__name__}: {e}. "
                    f"Released controller='{released}' last_action='{last_act}'."
                )
