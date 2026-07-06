import zmq
from datetime import datetime
import time
from threading import Thread, Event, Lock

from paradex.utils.system import get_pc_list, get_pc_ip

class remote_camera_controller:
    """Main-PC driver for a cluster of capture-PC camera daemons.

    Orchestrates the per-PC ``server_daemon`` processes over ZMQ REQ/REP:
    a background thread (:meth:`run`) opens a command socket per PC, then
    ticks a heartbeat loop that pushes ``start`` / ``stop`` / ``reload`` /
    ``end`` commands and folds each daemon's reply into a live health view
    (per-PC status plus per-camera frame-id stall detection). Public
    ``start`` / ``stop`` methods only flip :class:`threading.Event` objects
    that the loop consumes, so the caller never touches the sockets directly.

    Parameters
    ----------
    name : str
        Controller label; a ``_YYYYmmdd_HHMMSS`` timestamp is appended to make
        the unique ``controller_name`` sent with every command.
    pc_list : list of str, optional
        Capture PCs to drive. Defaults to ``get_pc_list()`` when ``None``.
    auto_reload : bool, optional
        If ``True``, reload the cameras automatically (throttled) whenever a
        persistent error or stall is detected, by default ``False``.
    stall_timeout : float, optional
        Seconds without a new frame id before a camera is flagged as stalled,
        by default 3.0.
    """

    def __init__(self, name, pc_list=None, auto_reload=False, stall_timeout=3.0):
        self.name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.pc_list = get_pc_list() if pc_list is None else pc_list

        self.ping_port = 5480
        self.monitor_port = 5481    # daemon PUB health telemetry (we SUB it)
        self.command_port = 5482

        self.exit_event = Event()
        self.start_event = Event()
        self.stop_event = Event()
        self.sending_event = Event()
        self.error_event = Event()
        self.init_error = None

        # ── health tracking ──────────────────────────────────────────────
        self.auto_reload = auto_reload          # auto-reload cameras on stall/error
        self.stall_timeout = stall_timeout      # s without a new frame → "stalled"
        self.capturing = False                  # in a continuous (video/stream/full) capture
        self.continuous = False
        self.cam_progress = {}                  # cam -> (last_frame_id, last_change_time)
        self.stalled = set()                    # cameras whose frame_id stopped advancing
        self.pc_status = {}                     # pc -> {'status', 'msg', ...} (from PUB)
        self.last_response = {}                 # last raw per-PC command reply (register/start/stop/reload)
        self.pc_last_seen = {}                  # pc -> monotonic time of last PUB health message
        self.pub_timeout = 2.0                  # s without any PUB from a PC → treat as down
        self.status_lock = Lock()
        self._last_stalled_print = set()
        self._last_reload_ts = 0.0
        self.health_thread = None
        self._reload_request = Event()          # health thread asks run() to reload

        self.run_thread = Thread(target=self.run, daemon=True)
        self.run_thread.start()

        
    def initialize(self):
        """Connect a command socket to every reachable capture PC and register.

        Pings each PC in ``pc_list``, opens a non-blocking ZMQ ``REQ`` command
        socket to the ones that answer, seeds the per-PC error/status tables,
        then calls :meth:`register` to claim the daemons. Runs once from
        :meth:`run` at thread start.

        Raises
        ------
        ConnectionError
            If any PC fails to respond to the ping (its ``server_daemon`` is
            not running).
        """
        self.ctx = zmq.Context()
        self.command_sockets = {}
        self.monitor_sockets = {}
        failed_pcs = []

        for pc in self.pc_list:
            if not self.check_server_alive(pc):
                failed_pcs.append(pc)
                continue

            socket = self.ctx.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 1000)
            socket.setsockopt(zmq.SNDTIMEO, 1000)
            socket.setsockopt(zmq.REQ_RELAXED, 1)
            socket.setsockopt(zmq.REQ_CORRELATE, 1)
            socket.connect(f"tcp://{get_pc_ip(pc)}:{self.command_port}")
            self.command_sockets[pc] = socket

            # SUB to the daemon's health telemetry (5481). Health is read from
            # here, not from the command reply, so a slow command on one PC can't
            # delay another PC's health/keepalive.
            msock = self.ctx.socket(zmq.SUB)
            msock.setsockopt(zmq.LINGER, 0)
            msock.setsockopt_string(zmq.SUBSCRIBE, '')
            msock.setsockopt(zmq.RCVHWM, 4)      # keep only the freshest few
            msock.setsockopt(zmq.CONFLATE, 0)    # (multipart json; conflate unsafe)
            msock.connect(f"tcp://{get_pc_ip(pc)}:{self.monitor_port}")
            self.monitor_sockets[pc] = msock
            print(f"{pc}: Command + health sockets connected")

        self.last_err = {pc: None for pc in self.pc_list}
        self.pc_status = {pc: {'status': None, 'msg': None} for pc in self.pc_list}
        self.pc_last_seen = {pc: 0.0 for pc in self.pc_list}

        if failed_pcs:
            raise ConnectionError(
                f"다음 PC들이 응답하지 않습니다: {failed_pcs}\n"
                f"각 PC에서 'python src/camera/server_daemon.py'를 실행하세요."
            )
        
        self.register()
    
    def check_server_alive(self, pc):
        """Ping a capture PC's daemon and report whether it is up.

        ping port로 서버 확인. Sends ``"ping"`` on the ping port and expects
        ``"pong"`` back within the socket timeout.

        Parameters
        ----------
        pc : str
            Capture-PC name to probe.

        Returns
        -------
        bool
            ``True`` if the daemon replied ``"pong"``, ``False`` on any ZMQ
            error or mismatched reply.
        """
        socket = self.ctx.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 5000)
        socket.setsockopt(zmq.SNDTIMEO, 5000)
        
        try:
            socket.connect(f"tcp://{get_pc_ip(pc)}:{self.ping_port}")
            socket.send_string("ping")
            response = socket.recv_string()
            return response == "pong"
        except zmq.ZMQError:
            return False
        finally:
            socket.close()
    
    def send_command(self, cmd):
        """Broadcast a command to every PC in parallel and collect replies.

        명령 전송 및 응답 수신 (PC 병렬 처리). Stamps ``cmd`` with this
        controller's name, then fans the request out over all command sockets
        on one thread each, joining before returning. ``start`` / ``stop``
        actions get a longer receive timeout than other actions. Socket-level
        failures are turned into an ``error`` status entry rather than raised.

        Parameters
        ----------
        cmd : dict
            Command payload with at least an ``action`` key; sent as JSON.

        Returns
        -------
        dict
            Maps each PC name to its JSON reply (or a synthesized
            ``{'status': 'error', ...}`` dict on timeout/ZMQ error).
        """
        cmd['controller_name'] = self.name
        response = {}
        response_lock = Lock()

        timeout_ms = 30000 if cmd.get('action') in ('start', 'stop') else 2000

        def _send_to_one(pc, socket):
            try:
                socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
                socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
                socket.send_json(cmd)
                resp = socket.recv_json()
            except zmq.Again:
                resp = {'status': 'error', 'msg': 'timeout', 'errno': 'EAGAIN'}
            except zmq.ZMQError as e:
                resp = {'status': 'error',
                        'msg': f'zmq:{e.errno}:{zmq.strerror(e.errno)}',
                        'errno': e.errno}
            with response_lock:
                response[pc] = resp

        threads = [Thread(target=_send_to_one, args=(pc, sock))
                   for pc, sock in self.command_sockets.items()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return response
            
    def register(self):
        """Claim the daemons for this controller by sending a ``register`` command."""
        cmd = {'action': 'register'}
        self.send_command(cmd)
        
    def start(self, mode, syncMode, save_path=None, fps=30, exposure_time=None, gain=None):
        """Begin a capture across all PCs; blocks until the command is sent.

        Stores the capture parameters, then signals :meth:`run` via
        ``start_event`` and waits on ``sending_event`` for the loop to actually
        dispatch the ``start`` command to the daemons (the handshake).

        Parameters
        ----------
        mode : str
            ``image`` / ``video`` / ``stream`` / ``full``.
        syncMode : bool
            Use the hardware trigger if ``True``.
        save_path : str, optional
            Output dir/file for ``video`` / ``image`` modes.
        fps : int, optional
            Frame rate for free-run capture, by default 30.
        exposure_time, gain : float, optional
            Per-capture overrides; ``None`` keeps each camera's baseline.
        """
        self.mode = mode
        self.syncMode = syncMode
        self.save_path = save_path
        self.fps = fps
        self.exposure_time = exposure_time
        self.gain = gain

        if self.init_error is not None:
            raise self.init_error

        self.sending_event.clear()
        self.start_event.set()

        self.sending_event.wait()
        if self.init_error is not None:
            raise self.init_error

    def stop(self):
        """Stop the current capture; blocks until the stop command is sent.

        Signals :meth:`run` via ``stop_event`` and waits on ``sending_event``
        for the loop to dispatch the ``stop`` command to every daemon.
        """
        if self.init_error is not None:
            raise self.init_error

        self.sending_event.clear()
        self.stop_event.set()
        self.sending_event.wait()
        if self.init_error is not None:
            raise self.init_error

    def end(self):
        """Shut down the controller and join the background thread.

        Sets ``exit_event`` so :meth:`run` breaks its loop (sending a final
        ``end`` command and closing the sockets/context), then joins the thread.
        """
        self.exit_event.set()
        self.run_thread.join()
        
    def reload_cameras(self):
        """Ask every daemon to reload its cameras; flag errors.

        Sends a ``reload`` command to all PCs and sets ``error_event`` for any
        PC whose reply reports an error.
        """
        cmd = {'action': 'reload'}
        response = self.send_command(cmd)
        
        for pc, resp in response.items():
            if resp['status'] == 'error':
                print(f"{pc}: {resp['msg']}")
                self.error_event.set()

    def force_takeover(self):
        """다른 controller 가 lock 잡고 있어도 강제로 register 재시도."""
        cmd = {'action': 'register', 'force': True}
        return self.send_command(cmd)

    def run(self):
        try:
            self.initialize()
        except Exception as e:
            self.init_error = e
            self.error_event.set()
            with self.status_lock:
                self.pc_status = {
                    pc: {'status': 'error', 'msg': str(e)}
                    for pc in self.pc_list
                }
            self.sending_event.set()
            return

        # Health is now read off the daemon PUB stream (5481) by this thread,
        # independent of the command channel below.
        self.health_thread = Thread(target=self._health_loop, daemon=True)
        self.health_thread.start()

        while not self.exit_event.is_set():
            # auto_reload asked for by the health thread — sent here so command
            # sockets are only ever touched by this thread.
            if self._reload_request.is_set():
                self._reload_request.clear()
                self.send_command({'action': 'reload'})

            cmd = {'action': 'heartbeat'}

            if self.start_event.is_set():
                cmd = {
                    'action': 'start',
                    'mode': self.mode,
                    'syncMode': self.syncMode,
                    'save_path': self.save_path,
                    'fps': self.fps,
                    'exposure_time': self.exposure_time,
                    'gain': self.gain
                }
                self.start_event.clear()

            if self.stop_event.is_set():
                cmd = {'action': 'stop'}
                self.stop_event.clear()

            response = self.send_command(cmd)

            # Command replies drive control flow only (did the daemon accept the
            # start/stop). Continuous health/stall/error now comes from _health_loop.
            if cmd['action'] != 'heartbeat':
                with self.status_lock:
                    self.last_response = {pc: dict(resp) for pc, resp in response.items()}

            if cmd['action'] == 'start':
                ok = all(resp.get('status') == 'ok' for resp in response.values())
                self.continuous = self.mode in ('video', 'stream', 'full')
                self.capturing = ok and self.continuous
                self.cam_progress = {}
            elif cmd['action'] == 'stop':
                self.capturing = False
                self.cam_progress = {}
                with self.status_lock:
                    self.stalled = set()

            if cmd['action'] in ('start', 'stop'):
                self.sending_event.set()
            time.sleep(0.1)

        self.send_command({'action': 'end'})
        # Stop the health thread before tearing down its sockets/context.
        if self.health_thread is not None:
            self.health_thread.join(timeout=2)
        for socket in self.command_sockets.values():
            socket.close()
        for socket in self.monitor_sockets.values():
            socket.close()
        self.ctx.term()

    def _health_loop(self):
        """Consume each daemon's PUB health (5481) and maintain the live view.

        Runs on its own thread, independent of the command channel, so a slow
        ``start`` / ``stop`` on one PC never delays another PC's health or stall
        detection. For each PC it drains the SUB socket to the freshest message,
        folds it into per-PC status + per-camera stall tracking, then recomputes
        the aggregate ``error_event`` — including PC liveness, so a daemon that
        stops publishing is treated as down.
        """
        while not self.exit_event.is_set():
            now = time.time()
            for pc, sock in self.monitor_sockets.items():
                latest = None
                while True:                       # drain to the newest message
                    try:
                        latest = sock.recv_json(flags=zmq.NOBLOCK)
                    except (zmq.Again, zmq.ZMQError):
                        break
                if latest is not None:
                    self._ingest_health(pc, latest, now)
            self._recompute_error(now)
            time.sleep(0.05)

    def _ingest_health(self, pc, msg, now):
        """Fold one PUB health message from ``pc`` into pc_status + stall tracking."""
        summary = msg.get('summary') or {}
        errors = summary.get('errors') or {}
        frame_ids = summary.get('frame_ids') or {}
        with self.status_lock:
            self.pc_last_seen[pc] = now
            self.pc_status[pc] = {
                'status': 'error' if errors else 'ok',
                'msg': (f"camera errors: {errors}" if errors else None),
                'states': dict(summary.get('states') or {}),
                'frame_ids': dict(frame_ids),
                'running': msg.get('running'),
                'controller': msg.get('controller'),
                'expected_camera_count': summary.get('expected_camera_count'),
                'detected_camera_count': summary.get('detected_camera_count'),
            }
            # Stall detection: only while a continuous capture should be flowing.
            if self.capturing and self.continuous:
                for cam, fid in frame_ids.items():
                    last = self.cam_progress.get(cam)
                    if last is None or fid != last[0]:
                        self.cam_progress[cam] = (fid, now)  # advanced → not stalled

    def _recompute_error(self, now):
        """Aggregate per-PC status, PC liveness and stalls into error_event (live)."""
        any_bad = False
        with self.status_lock:
            # A camera is stalled if its frame id hasn't advanced within the window.
            stalled = set()
            if self.capturing and self.continuous:
                for cam, (fid, ts) in self.cam_progress.items():
                    if now - ts > self.stall_timeout:
                        stalled.add(cam)
            self.stalled = stalled

            for pc in self.pc_list:
                # Liveness: a daemon that stopped publishing is down.
                if now - self.pc_last_seen.get(pc, 0.0) > self.pub_timeout:
                    any_bad = True
                    if self.last_err.get(pc) != 'no telemetry':
                        print(f"[{pc}] no health telemetry (daemon down / PUB silent)")
                        self.last_err[pc] = 'no telemetry'
                    continue
                st = self.pc_status.get(pc) or {}
                exp, det = st.get('expected_camera_count'), st.get('detected_camera_count')
                msg = st.get('msg')
                bad = (st.get('status') == 'error') or (
                    exp is not None and det is not None and det < exp)
                if bad:
                    any_bad = True
                    if self.last_err.get(pc) != msg:
                        print(f"[{pc}] {msg or 'camera count mismatch'}")
                        self.last_err[pc] = msg
                elif self.last_err.get(pc) is not None:
                    print(f"[{pc}] recovered after {self.last_err[pc]}")
                    self.last_err[pc] = None

        if stalled:
            any_bad = True
            if stalled != self._last_stalled_print:
                print(f"[stall] no new frames from: {sorted(stalled)} (> {self.stall_timeout}s)")
                self._last_stalled_print = set(stalled)
        else:
            self._last_stalled_print = set()

        # Live error flag (reflects the current tick, not a past blip).
        if any_bad:
            self.error_event.set()
        else:
            self.error_event.clear()

        # Optional self-heal: reload cameras on a persistent problem (throttled).
        # Routed through run() (the command-socket owner) via _reload_request.
        if self.auto_reload and any_bad and (now - self._last_reload_ts) > 10.0:
            self._last_reload_ts = now
            print("[auto_reload] problem detected → requesting camera reload")
            self._reload_request.set()
            with self.status_lock:
                self.cam_progress = {}

    def is_error(self):
        """True if any PC is currently erroring or any camera is stalled (live)."""
        return self.error_event.is_set()

    def get_status(self):
        """Live health snapshot for the caller to react to.

        Returns
        -------
        dict
            ``{'error': bool, 'stalled': [cam, ...], 'pc': {pc: {'status', 'msg'}}}``
        """
        with self.status_lock:
            return {
                'error': self.error_event.is_set(),
                'stalled': sorted(self.stalled),
                'pc': {pc: dict(s) for pc, s in self.pc_status.items()},
                'last_response': {pc: dict(resp) for pc, resp in self.last_response.items()},
            }
