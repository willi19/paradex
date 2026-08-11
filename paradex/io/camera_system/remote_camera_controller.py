import zmq
from datetime import datetime
import time
from threading import Thread, Event

from paradex.utils.system import get_pc_list, get_pc_ip

class remote_camera_controller:
    def __init__(self, name, pc_list=None, register_as_owner=True):
        self.name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.pc_list = get_pc_list() if pc_list is None else pc_list

        self.ping_port = 5480
        self.command_port = 5482
        self.connection_port = 5483

        # register_as_owner=False: a record-only side-channel client. It does
        # NOT register (so it can't steal the lock from the stream owner) and
        # does NOT heartbeat (the stream owner keeps the daemon alive). It only
        # emits record_start/record_stop on demand.
        self.register_as_owner = register_as_owner

        self.exit_event = Event()
        self.start_event = Event()
        self.stop_event = Event()
        self.record_start_event = Event()
        self.record_stop_event = Event()
        self.sending_event = Event()
        self.error_event = Event()

        self.record_save_path = None
        self.record_fps = 30

        self.run_thread = Thread(target=self.run, daemon=True)
        self.run_thread.start()

        
    def _new_command_socket(self, pc):
        """Create a fresh REQ socket bound to pc's command port.

        Pulled out so send_command can rebuild a broken socket: once a REQ
        socket times out without receiving its REP, it sits in EFSM
        (illegal state) and every subsequent send_json raises ZMQError. The
        only safe recovery is close + new socket. This is the actual cause
        of stream_owner getting stuck in error_event after recording — the
        daemon blocks ~5s flushing .avi files, the 1s heartbeat REQ times
        out, and the socket is dead from then on.
        """
        socket = self.ctx.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        # Heartbeat/command recv timeout. Raised 1000 -> 1500 ms: capturenew's
        # daemon occasionally answers a hair over 1 s, which tripped a false
        # "no response" and rebuilt the socket every time (connect/drop flapping).
        socket.setsockopt(zmq.RCVTIMEO, 1500)
        socket.setsockopt(zmq.SNDTIMEO, 1500)
        socket.connect(f"tcp://{get_pc_ip(pc)}:{self.command_port}")
        return socket

    def initialize(self):
        self.ctx = zmq.Context()
        self.command_sockets = {}
        failed_pcs = []

        for pc in self.pc_list:
            if not self.check_server_alive(pc):
                failed_pcs.append(pc)
                continue

            self.command_sockets[pc] = self._new_command_socket(pc)
            print(f"{pc}: Command socket connected")

        if failed_pcs:
            raise ConnectionError(
                f"다음 PC들이 응답하지 않습니다: {failed_pcs}\n"
                f"각 PC에서 'python src/camera/server_daemon.py'를 실행하세요."
            )

        if self.register_as_owner:
            self.register()
    
    def check_server_alive(self, pc):
        """ping port로 서버 확인"""
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
        """명령 전송 및 응답 수신.

        On any ZMQ error (send/recv timeout, EFSM after a previous timeout),
        the REQ socket is closed and rebuilt before reporting the error so
        the NEXT heartbeat can succeed. Without this, a single timeout —
        e.g. caused by the daemon blocking 5s on record_stop's AVI flush —
        wedges the socket in EFSM permanently and stream_owner sees endless
        "no response" until it restarts the entire stream.

        For record_start / record_stop the daemon synchronously waits for
        every camera's VideoWriter to release (up to flush_timeout=5s per
        camera), so the 1s heartbeat timeout will trip every time. We
        temporarily raise RCVTIMEO for those actions so the legitimate
        long flush isn't misreported as a failure.
        """
        cmd['controller_name'] = self.name
        response = {}

        # AVI flush on daemon side can legitimately take >1s; give it slack
        # before declaring a timeout.
        long_actions = {'record_start', 'record_stop', 'start', 'stop'}
        long_timeout_ms = 15000 if cmd.get('action') in long_actions else None

        # PARALLEL: fire the send to EVERY PC first, then collect the replies.
        # The daemons process concurrently and all receive the command within
        # ~ms of each other, instead of a slow PC gating the next one's send
        # (which offset the cameras' record start). Each PC has its own REQ
        # socket, so one send + one (later) recv per socket stays legal.
        pcs = list(self.command_sockets.keys())
        _limit_ms = long_timeout_ms if long_timeout_ms is not None else 1500
        old_rcv = {}
        t0 = {}
        sent = []

        for pc in pcs:
            socket = self.command_sockets[pc]
            if long_timeout_ms is not None:
                try:
                    old_rcv[pc] = socket.getsockopt(zmq.RCVTIMEO)
                    socket.setsockopt(zmq.RCVTIMEO, long_timeout_ms)
                except Exception:
                    old_rcv[pc] = None
            t0[pc] = time.perf_counter()
            try:
                socket.send_json(cmd)          # queues; daemons start working now
                sent.append(pc)
            except zmq.ZMQError as e:
                response[pc] = {'status': 'error', 'msg': f'no response (send: {e})'}
                print(f"[cam-timing] {pc} SEND failed on action='{cmd.get('action')}': {e}")
                self._rebuild_socket(pc)

        # Diagnostics: a timeout firing right at the limit => timeout too tight;
        # a reply well under the limit but >500ms => the daemon is lagging.
        for pc in sent:
            socket = self.command_sockets[pc]
            try:
                response[pc] = socket.recv_json()
            except zmq.ZMQError as e:
                _dt_ms = (time.perf_counter() - t0[pc]) * 1000.0
                response[pc] = {'status': 'error', 'msg': f'no response ({e})'}
                print(f"[cam-timing] {pc} TIMEOUT after {_dt_ms:.0f}ms "
                      f"(limit {_limit_ms}ms) on action='{cmd.get('action')}'")
                self._rebuild_socket(pc)
                continue
            _dt_ms = (time.perf_counter() - t0[pc]) * 1000.0
            if _dt_ms > 500:
                print(f"[cam-timing] {pc} slow reply {_dt_ms:.0f}ms "
                      f"(limit {_limit_ms}ms) on action='{cmd.get('action')}'")

        # Restore the short heartbeat timeout on the still-healthy sockets.
        if long_timeout_ms is not None:
            for pc in pcs:
                if old_rcv.get(pc) is not None and pc in self.command_sockets:
                    try:
                        self.command_sockets[pc].setsockopt(zmq.RCVTIMEO, old_rcv[pc])
                    except Exception:
                        pass

        return response

    def _rebuild_socket(self, pc):
        """Close + recreate a wedged REQ socket (it sits in EFSM after a timeout,
        so every later send raises until it is rebuilt)."""
        try:
            self.command_sockets[pc].close(linger=0)
        except Exception:
            pass
        try:
            self.command_sockets[pc] = self._new_command_socket(pc)
        except Exception as rebuild_err:
            print(f"{pc}: socket rebuild failed: {rebuild_err}")
            # Drop the broken entry; check_server_alive on next full restart
            # will recreate it.
            self.command_sockets.pop(pc, None)
            
    def register(self):
        cmd = {'action': 'register'}
        self.send_command(cmd)
        
    def start(self, mode, syncMode, save_path=None, fps=30):
        self.mode = mode
        self.syncMode = syncMode
        self.save_path = save_path
        self.fps = fps
        
        self.sending_event.clear()
        self.start_event.set()
        
        self.sending_event.wait()

    def stop(self):
        self.request_stop()
        self.wait_stopped()

    def request_stop(self):
        """Fire the camera stop (avi flush) without blocking. Pair with
        wait_stopped() so the ~flush overlaps other teardown (arm/hand save)."""
        self.sending_event.clear()
        self.stop_event.set()

    def wait_stopped(self, timeout=None):
        """Block until the async request_stop() has completed on the daemon."""
        self.sending_event.wait(timeout)

    def release(self):
        """Release the daemon controller lock without stopping cameras.

        Useful for viewer-style clients that want to disconnect/reconnect
        without tearing down an already-running stream.
        """
        response = self.send_command({'action': 'end'})
        for pc, resp in response.items():
            if resp['status'] == 'error':
                print(f"{pc}: {resp['msg']}")
                self.error_event.set()
        return response

    def record_start(self, save_path, fps=30):
        """Arm .avi recording on the (already-streaming) daemon. Non-blocking
        on acquisition: does not stop/restart cameras, stream is unaffected."""
        self.record_save_path = save_path
        self.record_fps = fps
        self.sending_event.clear()
        self.record_start_event.set()
        self.sending_event.wait()

    def record_stop(self):
        """Disarm .avi recording (stream keeps running)."""
        self.sending_event.clear()
        self.record_stop_event.set()
        self.sending_event.wait()

    def end(self):
        self.exit_event.set()        
        self.run_thread.join()
        
    def reload_cameras(self):
        cmd = {'action': 'reload'}
        response = self.send_command(cmd)
        
        for pc, resp in response.items():
            if resp['status'] == 'error':
                print(f"{pc}: {resp['msg']}")
                self.error_event.set()
        
    def run(self):
        self.initialize()
        
        while not self.exit_event.is_set():
            # Owner clients heartbeat to keep the daemon alive; record-only
            # side-channel clients stay silent unless toggling recording.
            cmd = {'action': 'heartbeat'} if self.register_as_owner else None

            if self.start_event.is_set():
                cmd = {
                    'action': 'start',
                    'mode': self.mode,
                    'syncMode': self.syncMode,
                    'save_path': self.save_path,
                    'fps': self.fps
                }
                self.start_event.clear()

            if self.stop_event.is_set():
                cmd = {'action': 'stop'}
                self.stop_event.clear()

            if self.record_start_event.is_set():
                cmd = {
                    'action': 'record_start',
                    'save_path': self.record_save_path,
                    'fps': self.record_fps,
                }
                self.record_start_event.clear()

            if self.record_stop_event.is_set():
                cmd = {'action': 'record_stop'}
                self.record_stop_event.clear()

            if cmd is None:
                time.sleep(0.1)
                continue

            response = self.send_command(cmd)
            if cmd['action'] in ['start', 'stop', 'record_start', 'record_stop']:
                self.sending_event.set()

            any_error = False
            dropped = False
            for pc, resp in response.items():
                if resp['status'] == 'error':
                    print(f"{pc}: {resp['msg']}")
                    any_error = True
                    if 'no active controller' in str(resp.get('msg', '')):
                        dropped = True
            # error_event reflects the *current* round, not stale failures —
            # one transient ZMQ timeout no longer sticks forever and trips
            # stream_owner's auto-restart.
            if any_error:
                self.error_event.set()
            else:
                self.error_event.clear()

            # The daemon's command socket has RCVTIMEO=5000, so any 5 s gap in
            # commands makes it call camera_loader.stop() and clear
            # current_controller (camera_server_daemon.command_thread). We only
            # registered inside initialize(), so without this every later
            # heartbeat answers "no active controller" forever: acquisition
            # stays dead, the shm freezes at all-zero, and this process still
            # looks healthy. Re-register from THIS thread — it owns the
            # sockets — and resume whatever mode we were running.
            if dropped and self.register_as_owner:
                print("[Info] daemon dropped the controller; re-registering")
                self.send_command({'action': 'register'})
                if getattr(self, 'mode', None) is not None:
                    self.start_event.set()

            time.sleep(0.1)

        if self.register_as_owner:
            self.send_command({'action': 'end'})
        for socket in self.command_sockets.values():
            socket.close()
        self.ctx.term()
    
    def is_error(self):
        return self.error_event.is_set()
