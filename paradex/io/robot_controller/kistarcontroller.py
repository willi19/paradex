import ctypes
import os
import time
from threading import Event, Lock, Thread
from typing import Optional, Sequence

import numpy as np


HAND_DOF = 16
KIN_SENSOR_NUM = 4
KIN_SENSOR_DOF = 3
TACTILE_NUM = 60
SHM_MSG_KEY = 0x2969
PERMISSIONS = 0o666

# Same per-joint ranges used by KISTAR_HAND.py
JOINT_LIMITS = [
    (0, 4096),
    (-4096, 4096),
    (0, 4096),
    (0, 4096),
    (-680, 680),
    (0, 4096),
    (0, 4096),
    (0, 4096),
    (-680, 680),
    (0, 4096),
    (0, 4096),
    (0, 4096),
    (-680, 680),
    (0, 4096),
    (0, 4096),
    (0, 4096),
]


class SHMmsgs(ctypes.Structure):
    _fields_ = [
        ("Status1", ctypes.c_uint16),
        ("Status2", ctypes.c_uint16),
        ("j_pos", ctypes.c_int16 * HAND_DOF),
        ("j_tar", ctypes.c_int16 * HAND_DOF),
        ("j_cur", ctypes.c_int16 * HAND_DOF),
        ("j_kin", (ctypes.c_int16 * KIN_SENSOR_DOF) * KIN_SENSOR_NUM),
        ("j_tac", ctypes.c_int16 * TACTILE_NUM),
        ("interpolation_duration", ctypes.c_int16),
        ("process_num", ctypes.c_int32),
    ]


class SharedMemoryClient:
    def __init__(self):
        self.libc = ctypes.CDLL("libc.so.6", use_errno=True)
        self.libc.shmget.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_int]
        self.libc.shmget.restype = ctypes.c_int
        self.libc.shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self.libc.shmat.restype = ctypes.c_void_p
        self.libc.shmdt.argtypes = [ctypes.c_void_p]
        self.libc.shmdt.restype = ctypes.c_int
        self.shm_id = -1
        self.shm_ptr = None
        self.msgs = None

    def connect(self):
        self.shm_id = self.libc.shmget(SHM_MSG_KEY, ctypes.sizeof(SHMmsgs), PERMISSIONS)
        if self.shm_id < 0:
            err = ctypes.get_errno()
            raise OSError(
                err,
                f"shmget failed (key=0x{SHM_MSG_KEY:x}). Run KISTAR_TEST first to create SHM.",
            )
        addr = self.libc.shmat(self.shm_id, None, 0)
        if addr == ctypes.c_void_p(-1).value:
            err = ctypes.get_errno()
            raise OSError(err, "shmat failed")
        self.shm_ptr = addr
        self.msgs = ctypes.cast(addr, ctypes.POINTER(SHMmsgs))

    def close(self):
        if self.shm_ptr:
            self.libc.shmdt(ctypes.c_void_p(self.shm_ptr))
            self.shm_ptr = None
            self.msgs = None


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


class KistarController:
    """
    KISTAR SHM controller with AllegroController-like lifecycle:
    - move(action) updates desired action buffer
    - background loop writes j_tar at fixed rate
    - start/stop records action and SHM states
    """

    def __init__(
        self,
        rate_hz: float = 60.0,
        interp_ms: int = 100,
        servo_on: bool = True,
        position_mode: bool = True,
        clamp_action: bool = True,
        home_from: str = "pos",
        joint_map: Sequence[int] = (0, 1, 2, 3, 4, 5),
        gain: Sequence[float] = (450, 450, 450, 450, 220, 220),
        sign: Sequence[float] = (1, 1, 1, 1, 1, 1),
    ):
        if home_from not in ("pos", "tar"):
            raise ValueError("home_from must be 'pos' or 'tar'")
        if len(joint_map) != 6 or len(gain) != 6 or len(sign) != 6:
            raise ValueError("joint_map, gain, sign must each have 6 values")

        self.rate_hz = float(max(1e-3, rate_hz))
        self.home_from = home_from
        self.clamp_action = bool(clamp_action)
        self.joint_map = [int(v) for v in joint_map]
        self.gain = np.asarray(gain, dtype=np.float64)
        self.sign = np.asarray(sign, dtype=np.float64)

        for j in self.joint_map:
            if j < 0 or j >= HAND_DOF:
                raise ValueError(f"joint_map has invalid index: {j}")

        self.lock = Lock()
        self.save_event = Event()
        self.exit_event = Event()
        self.connection_event = Event()
        self.error_event = Event()

        self.capture_path = None
        self.data = None

        self.action = None
        self.ref_action = None
        self.base_cmd = None
        self.cmd = np.zeros(HAND_DOF, dtype=np.int32)
        self.last_input_action = None

        self.shm = SharedMemoryClient()
        self.shm.connect()

        m = self.shm.msgs.contents
        if servo_on:
            m.Status1 = 0xFFFF
        if position_mode:
            m.Status2 = 1
        m.interpolation_duration = int(interp_ms)

        home = np.array(m.j_pos if self.home_from == "pos" else m.j_tar, dtype=np.int32)
        self.base_cmd = home.copy()
        self.cmd = home.copy()
        self.action = self.cmd.copy()
        self.connection_event.set()

        self.thread = Thread(target=self.control_loop, daemon=True)
        self.thread.start()

    def _apply_direct16(self, action_arr: np.ndarray):
        for i in range(HAND_DOF):
            target = int(np.rint(action_arr[i]))
            if self.clamp_action:
                lo, hi = JOINT_LIMITS[i]
                target = _clamp(target, lo, hi)
            self.cmd[i] = target

    def _resolve_action_to_cmd(self, action_arr: np.ndarray):
        if action_arr.size == HAND_DOF:
            self._apply_direct16(action_arr)
            return

        raise ValueError(f"Unsupported action size: {action_arr.size} (expected 16)")

    def _publish_cmd(self):
        m = self.shm.msgs.contents
        for i in range(HAND_DOF):
            m.j_tar[i] = int(self.cmd[i])

    def _record_if_needed(self):
        if not self.save_event.is_set() or self.data is None:
            return
        m = self.shm.msgs.contents
        self.data["time"].append(time.time())
        self.data["j_tar"].append(np.array(m.j_tar, dtype=np.int32))
        self.data["j_pos"].append(np.array(m.j_pos, dtype=np.int32))
        self.data["j_tac"].append(np.array(m.j_tac, dtype=np.int32))

    def control_loop(self):
        while not self.connection_event.is_set() and not self.exit_event.is_set():
            time.sleep(0.01)

        dt = 1.0 / self.rate_hz
        while not self.exit_event.is_set():
            t0 = time.perf_counter()
            try:
                with self.lock:
                    desired = None if self.action is None else self.action.copy()
                    if desired is not None:
                        self._resolve_action_to_cmd(desired)
                        self._publish_cmd()
                    self._record_if_needed()
            except Exception:
                self.error_event.set()

            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def start(self, save_path: str):
        self.capture_path = save_path
        self.data = {
            "time": [],
            "j_tar": [],
            "j_pos": [],
            "j_tac": [],
        }
        self.save_event.set()

    def stop(self):
        self.save_event.clear()
        if self.capture_path is None or self.data is None:
            return

        os.makedirs(self.capture_path, exist_ok=True)
        np.save(os.path.join(self.capture_path, "time.npy"), np.array(self.data["time"], dtype=np.float64))
        np.save(os.path.join(self.capture_path, "action.npy"), np.array(self.data["j_tar"], dtype=np.int32))
        np.save(os.path.join(self.capture_path, "qpos.npy"), np.array(self.data["j_pos"], dtype=np.int32))
        np.save(os.path.join(self.capture_path, "tactile.npy"), np.array(self.data["j_tac"], dtype=np.int32))
        self.capture_path = None
        self.data = None

    def end(self):
        self.exit_event.set()
        self.thread.join(timeout=1.0)
        if self.save_event.is_set():
            self.stop()
        self.shm.close()

    def move(self, action):
        if action is None:
            return
        try:
            action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        except Exception:
            self.error_event.set()
            return

        with self.lock:
            self.last_input_action = action_arr.copy()
            self.action = action_arr.copy()

    def get_data(self):
        with self.lock:
            m = self.shm.msgs.contents
            return {
                "qpos": np.array(m.j_pos, dtype=np.int32),
                "action": np.array(m.j_tar, dtype=np.int32),
                "tactile": np.array(m.j_tac, dtype=np.int32),
                "current": np.array(m.j_cur, dtype=np.int32),
                "time": time.time(),
            }

    def is_error(self):
        return self.error_event.is_set()
