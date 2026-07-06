import os
import threading
import zmq
import time
import traceback

from paradex.io.camera_system.camera_loader import CameraLoader
from paradex.io.camera_system.protocol import PROTOCOL_VERSION, get_auth_token
from paradex.utils.log import get_logger

logger = get_logger("camera")

class camera_server_daemon:
    def __init__(self, idle_timeout_s=None):
        self.camera_loader = CameraLoader()

        # Optional shared secret; None = accept any peer (closed-LAN default).
        self._auth_token = get_auth_token()
        if self._auth_token is not None:
            logger.info("command auth token required (PARADEX_CAMERA_TOKEN set)")

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
        logger.info("Camera loader reloaded.")

    def shutdown(self):
        """Release every camera (DeInit + free SHM) on a clean exit.

        Call from a SIGTERM/SIGINT handler so a normal kill releases the hardware;
        SIGKILL (-9) cannot be caught (next daemon start cleans up leaked SHM)."""
        logger.info("Shutting down: releasing cameras...")
        try:
            self.camera_loader.end()
        except Exception as e:
            logger.warning(f"shutdown: camera_loader.end() failed: {e}")

    def pingpong_thread(self):
        self.ping_socket = self.ctx.socket(zmq.REP)
        self.ping_socket.setsockopt(zmq.LINGER, 0)
        self.ping_socket.bind(f"tcp://*:{self.ping_port}")

        while True:
            try:
                _ = self.ping_socket.recv_string(flags=zmq.NOBLOCK)
                self.ping_socket.send_string("pong")
            except zmq.Again:
                time.sleep(0.1)          # no ping waiting — normal
            except Exception as e:
                logger.warning(f"pingpong_thread error (continuing): {e}")
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
            # Keep publishing even if a snapshot build hiccups — if this thread
            # died, the controller would (correctly but misleadingly) read the whole
            # daemon as "down" from the PUB silence.
            try:
                status = {
                    # dashboard-facing (unchanged keys, kept for back-compat)
                    'cameras': self.camera_loader.get_status_list(),
                    'controller': self.current_controller if self.current_controller else 'None',
                    # controller-facing health telemetry
                    'running': self.cameras_running,
                    'summary': self.camera_loader.get_summary(),
                }
                monitor_socket.send_json(status)
            except Exception as e:
                logger.warning(f"monitor_thread iteration failed (continuing): {e}")
            time.sleep(0.1)

    def execute_command(self, cmd):
        action = cmd.get('action')
        controller_name = cmd.get('controller_name')
        force = cmd.get('force', False)

        # Authentication: reject any peer without the shared token (when one is set).
        if self._auth_token is not None and cmd.get('token') != self._auth_token:
            logger.warning(f"rejected {action} from '{controller_name}': bad/missing auth token")
            return {"status": "error", "msg": "auth failed"}

        if action == "register":
            prev = self.current_controller
            # Real lock: refuse to steal a live controller's session unless forced
            # (force_takeover). A daemon that was just restarted has prev=None, and
            # the idle dead-man clears prev, so a legitimate (re)claim still works.
            if prev is not None and prev != controller_name and not force:
                logger.warning(f"register from '{controller_name}' refused: locked by '{prev}'")
                return {"status": "error", "msg": f"locked by {prev}; use force_takeover"}
            # Only clear a running capture on an actual takeover (a *different*
            # controller claiming the lock). A same-controller re-register — the
            # recovery path when one PC's daemon restarted — must NOT stop capture
            # on the healthy PCs that also receive the broadcast register.
            takeover = prev is not None and prev != controller_name
            if self.cameras_running and takeover:
                try:
                    self.camera_loader.stop()
                except Exception as e:
                    logger.warning(f"stop on register failed: {e}")
                self.cameras_running = False
            self.current_controller = controller_name
            self.last_action = action
            self.last_action_time = time.time()
            if prev and prev != controller_name:
                logger.info(f"controller takeover: '{prev}' -> '{controller_name}' (force={force})")
            resp = {"status": "ok", "msg": "registered", "version": PROTOCOL_VERSION}
            client_ver = cmd.get("version")
            if client_ver is not None and client_ver != PROTOCOL_VERSION:
                warn = f"protocol mismatch: controller={client_ver} daemon={PROTOCOL_VERSION} (git out of sync?)"
                logger.warning(warn)
                resp["warning"] = warn
            return resp

        if controller_name != self.current_controller and self.current_controller is not None:
            logger.warning(f"{controller_name} tried to access, but locked by {self.current_controller}")
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
                    logger.warning(f"start: {len(errs)} camera(s) failed to arm: {detail}")
                    payload.update({"status": "error", "msg": f"start: camera errors: {detail}"})
                    return payload
                # "acquire" = armed continuously (sinks toggled via the 'sink'
                # command); a running capture. "image" is a one-shot, not running.
                self.cameras_running = cmd.get('mode') == "acquire"
                payload["running"] = self.cameras_running
                logger.info(f"start completed in {dt:.2f}s mode={cmd.get('mode')} sync={cmd.get('syncMode')} fps={cmd.get('fps',30)} exposure_time={cmd.get('exposure_time')} gain={cmd.get('gain')}")
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
                logger.info(f"stop completed in {dt:.2f}s")
                payload = self.camera_loader.get_summary()
                payload.update({"status":"ok", "msg":"stopped", "running": False})
                return payload
            except Exception as e:
                dt = time.time() - t0
                traceback.print_exc()
                payload = self.camera_loader.get_summary()
                payload.update({"status":"error", "msg":f"stop failed after {dt:.2f}s: {type(e).__name__}:{e}", "running": self.cameras_running})
                return payload

        if action == "sink":
            # Toggle output sinks (video / stream / snapshot) on a running capture
            # without restarting acquisition. Requires an armed capture.
            try:
                self.camera_loader.set_sink(
                    video=cmd.get('video'),
                    stream=cmd.get('stream'),
                    save_path=cmd.get('save_path'),
                    snapshot=cmd.get('snapshot'),
                )
                payload = self.camera_loader.get_summary()
                payload.update({"status": "ok", "msg": "sink set", "running": self.cameras_running})
                return payload
            except Exception as e:
                traceback.print_exc()
                payload = self.camera_loader.get_summary()
                payload.update({"status": "error", "msg": f"sink failed: {type(e).__name__}:{e}",
                                "running": self.cameras_running})
                return payload

        if action == "param":
            # Live gain/exposure change on a running capture (no restart).
            try:
                self.camera_loader.set_param(gain=cmd.get('gain'), exposure=cmd.get('exposure'))
                payload = self.camera_loader.get_summary()
                payload.update({"status": "ok", "msg": "param set", "running": self.cameras_running})
                return payload
            except Exception as e:
                traceback.print_exc()
                payload = self.camera_loader.get_summary()
                payload.update({"status": "error", "msg": f"param failed: {type(e).__name__}:{e}",
                                "running": self.cameras_running})
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
            # Health/errors/frame-ids now travel on the PUB channel (5481), so the
            # heartbeat is just a cheap keepalive that resets the dead-man timer —
            # no get_summary()/get_all_errors() on the keepalive path.
            return {"status": "ok", "msg": "heartbeat", "running": self.cameras_running}

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
                    logger.info(
                        f"Idle timeout (>{self.idle_timeout_s}s, actual={idle:.1f}s): "
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
                logger.error(
                    f"Exception in command thread: {type(e).__name__}: {e}. "
                    f"Released controller='{released}' last_action='{last_act}'."
                )
