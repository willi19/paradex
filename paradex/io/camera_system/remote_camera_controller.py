import zmq
from datetime import datetime
import time
from threading import Thread, Event, Lock

from paradex.utils.system import get_pc_list, get_pc_ip

class remote_camera_controller:
    def __init__(self, name, pc_list=None, auto_reload=False, stall_timeout=3.0):
        self.name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.pc_list = get_pc_list() if pc_list is None else pc_list

        self.ping_port = 5480
        self.command_port = 5482

        self.exit_event = Event()
        self.start_event = Event()
        self.stop_event = Event()
        self.sending_event = Event()
        self.error_event = Event()

        # ── health tracking ──────────────────────────────────────────────
        self.auto_reload = auto_reload          # auto-reload cameras on stall/error
        self.stall_timeout = stall_timeout      # s without a new frame → "stalled"
        self.capturing = False                  # in a continuous (video/stream/full) capture
        self.continuous = False
        self.cam_progress = {}                  # cam -> (last_frame_id, last_change_time)
        self.stalled = set()                    # cameras whose frame_id stopped advancing
        self.pc_status = {}                     # pc -> {'status', 'msg'} (last heartbeat)
        self.status_lock = Lock()
        self._last_stalled_print = set()
        self._last_reload_ts = 0.0

        self.run_thread = Thread(target=self.run, daemon=True)
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
            socket.setsockopt(zmq.RCVTIMEO, 1000)
            socket.setsockopt(zmq.SNDTIMEO, 1000)
            socket.setsockopt(zmq.REQ_RELAXED, 1)
            socket.setsockopt(zmq.REQ_CORRELATE, 1)
            socket.connect(f"tcp://{get_pc_ip(pc)}:{self.command_port}")
            self.command_sockets[pc] = socket
            print(f"{pc}: Command socket connected")

        self.last_err = {pc: None for pc in self.pc_list}
        self.pc_status = {pc: {'status': None, 'msg': None} for pc in self.pc_list}

        if failed_pcs:
            raise ConnectionError(
                f"다음 PC들이 응답하지 않습니다: {failed_pcs}\n"
                f"각 PC에서 'python src/camera/server_daemon.py'를 실행하세요."
            )
        
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
        """명령 전송 및 응답 수신 (PC 병렬 처리)"""
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
        cmd = {'action': 'register'}
        self.send_command(cmd)
        
    def start(self, mode, syncMode, save_path=None, fps=30, exposure_time=None, gain=None):
        self.mode = mode
        self.syncMode = syncMode
        self.save_path = save_path
        self.fps = fps
        self.exposure_time = exposure_time
        self.gain = gain

        self.sending_event.clear()
        self.start_event.set()

        self.sending_event.wait()

    def stop(self):
        self.sending_event.clear()
        self.stop_event.set()
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

    def force_takeover(self):
        """다른 controller 가 lock 잡고 있어도 강제로 register 재시도."""
        cmd = {'action': 'register', 'force': True}
        return self.send_command(cmd)

    def run(self):
        self.initialize()

        while not self.exit_event.is_set():
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

            # Track whether we should be seeing a continuous frame stream.
            if cmd['action'] == 'start':
                self.continuous = self.mode in ('video', 'stream', 'full')
                self.capturing = self.continuous
                self.cam_progress = {}
            elif cmd['action'] == 'stop':
                self.capturing = False

            if cmd['action'] in ('start', 'stop'):
                self.sending_event.set()

            self._update_health(cmd['action'], response)
            time.sleep(0.1)

        self.send_command({'action': 'end'})
        for socket in self.command_sockets.values():
            socket.close()
        self.ctx.term()

    def _update_health(self, action, response):
        """Update live per-PC status, detect frame stalls, and set/clear error_event.

        Runs every loop tick from run(). error_event reflects the CURRENT state
        (not sticky). Frame stalls are detected from the per-camera frame_ids the
        daemon returns in heartbeat responses."""
        now = time.time()
        any_bad = False
        stalled = set()

        with self.status_lock:
            for pc, resp in response.items():
                self.pc_status[pc] = {'status': resp.get('status'), 'msg': resp.get('msg')}

                if resp.get('status') == 'error':
                    any_bad = True
                    msg = resp.get('msg')
                    if self.last_err.get(pc) != msg:
                        print(f"[{pc}] {action} failed: {msg}")
                        self.last_err[pc] = msg
                else:
                    if self.last_err.get(pc) is not None:
                        print(f"[{pc}] recovered after {self.last_err[pc]}")
                        self.last_err[pc] = None

                # Stall detection: only while a continuous capture should be flowing.
                if self.capturing and self.continuous:
                    for cam, fid in (resp.get('frame_ids') or {}).items():
                        last = self.cam_progress.get(cam)
                        if last is None or fid != last[0]:
                            self.cam_progress[cam] = (fid, now)
                        elif now - last[1] > self.stall_timeout:
                            stalled.add(cam)

            self.stalled = stalled

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
        if self.auto_reload and any_bad and (now - self._last_reload_ts) > 10.0:
            self._last_reload_ts = now
            print("[auto_reload] problem detected → reloading cameras")
            resp = self.send_command({'action': 'reload'})
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
            }