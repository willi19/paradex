import time
import os
from threading import Thread, Event, Lock
from typing import Optional, List

import numpy as np
import serial

# RH56F1 RS485 register map (from inspire-hand-ros2)
ADDR_ID = 0x03E8
ADDR_BAUD = 0x03EA
ADDR_CLEAR_ERROR = 0x03EC  # may be unsupported on some firmware

ADDR_SET_ANGLE = 0x0410
ADDR_SET_FORCE = 0x0416
ADDR_SET_SPEED = 0x041C
ADDR_ACTUAL_ANGLE = 0x0428
ADDR_ACTUAL_FORCE = 0x042E
ADDR_ACTUATOR_CURRENT = 0x0434
ADDR_MOTION_MODE = 0x044C
ADDR_TACTILE = 0x0BB8
TACTILE_LEN_BYTES = 68  # 34 registers

ACTION_DOF = 6

TACTILE_LABELS = [
    "little_normal_force",
    "little_tangential_force",
    "little_tangential_direction",
    "little_tactile_proximity",
    "ring_normal_force",
    "ring_tangential_force",
    "ring_tangential_direction",
    "ring_tactile_proximity",
    "middle_normal_force",
    "middle_tangential_force",
    "middle_tangential_direction",
    "middle_tactile_proximity",
    "index_normal_force",
    "index_tangential_force",
    "index_tangential_direction",
    "index_tactile_proximity",
    "thumb_normal_force",
    "thumb_tangential_force",
    "thumb_tangential_direction",
    "thumb_tactile_proximity",
    "palm_right_normal_force",
    "palm_right_tangential_force",
    "palm_right_tangential_direction",
    "palm_middle_normal_force",
    "palm_middle_tangential_force",
    "palm_middle_tangential_direction",
    "palm_left_normal_force",
    "palm_left_tangential_force",
    "palm_left_tangential_direction",
]


def _checksum_excluding_header(frame_wo_header: bytes) -> int:
    return sum(frame_wo_header) & 0xFF


def _build_read_frame(hand_id: int, addr: int, reg_len_bytes: int) -> bytes:
    body = bytearray([
        hand_id & 0xFF,
        0x04,
        0x11,
        addr & 0xFF,
        (addr >> 8) & 0xFF,
        reg_len_bytes & 0xFF,
    ])
    cs = _checksum_excluding_header(body)
    return bytes([0xEB, 0x90]) + bytes(body) + bytes([cs])


def _build_write_frame(hand_id: int, addr: int, payload: bytes) -> bytes:
    length = len(payload) + 3
    body = bytearray([
        hand_id & 0xFF,
        length & 0xFF,
        0x12,
        addr & 0xFF,
        (addr >> 8) & 0xFF,
    ])
    body.extend(payload)
    cs = _checksum_excluding_header(body)
    return bytes([0xEB, 0x90]) + bytes(body) + bytes([cs])


def _read_frame(ser: serial.Serial, timeout_s: float = 0.5) -> Optional[bytes]:
    t0 = time.time()
    buf = bytearray()
    while time.time() - t0 < timeout_s:
        b = ser.read(1)
        if not b:
            continue
        buf += b
        if len(buf) >= 2 and buf[-2] == 0x90 and buf[-1] == 0xEB:
            break
    else:
        return None

    hdr = ser.read(2)
    if len(hdr) < 2:
        return None
    data_len = hdr[1]
    rest = ser.read(data_len + 1)
    if len(rest) < data_len + 1:
        return None
    return bytes([0x90, 0xEB, hdr[0], data_len]) + rest


def _parse_read_resp(frame: bytes) -> Optional[bytes]:
    if len(frame) < 9:
        return None
    if not (frame[0] == 0x90 and frame[1] == 0xEB and frame[4] == 0x11):
        return None
    data_len = frame[3]
    reg_len = data_len - 3
    return frame[7:7 + reg_len]


def _u16_le(x: int) -> bytes:
    x &= 0xFFFF
    return bytes([x & 0xFF, (x >> 8) & 0xFF])


def _i16_from_le(b: bytes) -> int:
    return int.from_bytes(b, byteorder="little", signed=True)


def _to_raw_targets(action: np.ndarray) -> List[int]:
    # action: [0..1000] per DOF, open=1000, close=0
    # raw units are 0.1 deg with offsets.
    if action.shape != (ACTION_DOF,):
        raise ValueError("action must be shape (6,)")
    out = []
    limits = [1740, 1740, 1740, 1740, 1350, 1800]
    for i in range(ACTION_DOF):
        v = action[i]
        if v < 0:
            out.append(-1)
            continue
        v = float(v)
        v = max(0.0, min(1000.0, v))
        out.append(int(limits[i] * v / 1000.0))
    return out


def _raw_to_qpos(raw: np.ndarray) -> np.ndarray:
    # raw units: 0.1 deg
    # map to radians per rh56f1_hardware.cpp
    raw = raw.astype(np.float64)
    hw_angle = raw * 0.1
    qpos = np.zeros_like(hw_angle)
    for i in range(ACTION_DOF):
        if i < 4:
            qpos[i] = (174.0 - hw_angle[i]) / 180.0 * np.pi
        elif i == 4:
            qpos[i] = (135.0 - hw_angle[i]) / 180.0 * np.pi
        else:
            qpos[i] = (180.0 - hw_angle[i]) / 180.0 * np.pi
    return qpos


def _decode_tactile(raw_regs: np.ndarray) -> np.ndarray:
    """
    Decode RH56F1 tactile registers into 29-element state vector:
    - 5 fingers * (normal, tangential, direction, proximity) = 20
    - 3 palms * (normal, tangential, direction) = 9
    """
    if raw_regs is None or raw_regs.size < 34:
        return np.zeros(29, dtype=np.float64)

    states = np.zeros(29, dtype=np.float64)
    for idx in range(5):
        base = 5 * idx
        states[4 * idx + 0] = float(raw_regs[base + 0]) * 0.01
        states[4 * idx + 1] = float(raw_regs[base + 1]) * 0.01
        states[4 * idx + 2] = float(raw_regs[base + 2])
        proximity_low = int(raw_regs[base + 3]) & 0xFFFF
        proximity_high = int(raw_regs[base + 4]) & 0xFF
        proximity_24 = proximity_low | (proximity_high << 16)
        states[4 * idx + 3] = float(proximity_24 & 0xFFFFFF)
    for idx in range(3):
        base = 25 + 3 * idx
        states[20 + 3 * idx + 0] = float(raw_regs[base + 0]) * 0.01
        states[20 + 3 * idx + 1] = float(raw_regs[base + 1]) * 0.01
        states[20 + 3 * idx + 2] = float(raw_regs[base + 2])
    return states


class InspireF1Controller:
    def __init__(self, port: str, hand_id: int = 1, baud_rate: int = 115200, tactile: bool = False):
        self.port = port
        self.hand_id = hand_id
        self.baud_rate = baud_rate
        self.tactile = tactile

        self.lock = Lock()
        self.serial_lock = Lock()
        self.exit_event = Event()
        self.save_event = Event()
        self.error_event = Event()
        self.connection_event = Event()

        self.action = np.zeros(ACTION_DOF, dtype=np.float64) + 1000
        self.latest_qpos = None
        self.latest_force = None
        self.latest_current = None
        self.latest_tactile_raw = None
        self.latest_tactile = None
        self.latest_time = None

        self.save_path = None
        self.data = None

        self.ser = serial.Serial(self.port, self.baud_rate, timeout=0.05)
        self._init_device()

        self.control_thread = Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()
        self.connection_event.set()

    def _init_device(self):
        # try clear error and set motion mode + defaults
        try:
            self._write_raw(ADDR_CLEAR_ERROR, bytes([1]))
        except Exception:
            pass
        self._write_group_6(ADDR_MOTION_MODE, [0] * ACTION_DOF)
        self._write_group_6(ADDR_SET_SPEED, [1000] * ACTION_DOF)
        self._write_group_6(ADDR_SET_FORCE, [300] * ACTION_DOF)

    def _write_raw(self, addr: int, payload: bytes) -> Optional[int]:
        with self.serial_lock:
            self.ser.reset_input_buffer()
            frame = _build_write_frame(self.hand_id, addr, payload)
            self.ser.write(frame)
            resp = _read_frame(self.ser, 0.5)
            if resp is None:
                return None
            # return ack byte if present
            if len(resp) == 9 and resp[4] == 0x12:
                return resp[7]
            return None

    def _read_raw(self, addr: int, nbytes: int) -> Optional[bytes]:
        with self.serial_lock:
            self.ser.reset_input_buffer()
            frame = _build_read_frame(self.hand_id, addr, nbytes)
            self.ser.write(frame)
            resp = _read_frame(self.ser, 0.5)
            if resp is None:
                return None
            return _parse_read_resp(resp)

    def _write_group_6(self, addr: int, values: List[int]):
        payload = b"".join(_u16_le(v) for v in values)
        self._write_raw(addr, payload)

    def _read_group_6(self, addr: int) -> Optional[np.ndarray]:
        data = self._read_raw(addr, 12)
        if data is None or len(data) != 12:
            return None
        out = np.zeros(ACTION_DOF, dtype=np.int16)
        for i in range(ACTION_DOF):
            out[i] = _i16_from_le(data[i * 2:i * 2 + 2])
        return out

    def _read_tactile_raw(self) -> Optional[np.ndarray]:
        data = self._read_raw(ADDR_TACTILE, TACTILE_LEN_BYTES)
        if data is None or len(data) != TACTILE_LEN_BYTES:
            return None
        regs = np.zeros(34, dtype=np.uint16)
        for i in range(34):
            regs[i] = int.from_bytes(data[i * 2:i * 2 + 2], byteorder="little", signed=False)
        return regs

    def control_loop(self):
        self.fps = 30
        while not self.exit_event.is_set():
            start_time = time.time()
            with self.lock:
                action = self.action.copy()

            try:


                raw_angle = self._read_group_6(ADDR_ACTUAL_ANGLE)
                raw_force = self._read_group_6(ADDR_ACTUAL_FORCE)
                raw_current = self._read_group_6(ADDR_ACTUATOR_CURRENT)
                raw_tactile = self._read_tactile_raw() if self.tactile else None
                
                # print(f"Raw Angle: {raw_angle}")
                # print(f"Raw action: {action}")
                
                if raw_angle[5] < 1100:
                   action[:4] = np.clip(action[:4], 800, 1000)
                if raw_angle[0] < 1392 or raw_angle[1] < 1392 or raw_angle[2] < 1392 or raw_angle[3] < 1392:
                    action[5] = np.clip(action[5], 640, 1000)
                raw_targets = _to_raw_targets(action)
                self._write_group_6(ADDR_SET_ANGLE, raw_targets)
                    

                if raw_angle is not None:
                    qpos = _raw_to_qpos(raw_angle.astype(np.float64))
                    self.latest_qpos = qpos
                if raw_force is not None:
                    self.latest_force = raw_force.astype(np.float64)
                if raw_current is not None:
                    self.latest_current = raw_current.astype(np.float64)
                if raw_tactile is not None:
                    self.latest_tactile_raw = raw_tactile
                    self.latest_tactile = _decode_tactile(raw_tactile.astype(np.float64))
                self.latest_time = time.time()

                if self.save_event.is_set() and self.data is not None:
                    self.data["time"].append(self.latest_time)
                    if self.latest_qpos is not None:
                        self.data["position"].append(self.latest_qpos.copy())
                    if self.latest_force is not None:
                        self.data["force"].append(self.latest_force.copy())
                    if self.latest_current is not None:
                        self.data["current"].append(self.latest_current.copy())
                    if self.latest_tactile_raw is not None:
                        self.data["tactile_raw"].append(self.latest_tactile_raw.copy())
                    if self.latest_tactile is not None:
                        self.data["tactile"].append(self.latest_tactile.copy())
                    self.data["action"].append(action.copy())
            except Exception:
                self.error_event.set()

            elapsed = time.time() - start_time
            time.sleep(max(0.0, (1.0 / self.fps) - elapsed))

    def start(self, save_path: str):
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        with self.lock:
            self.data = {
                "time": [],
                "position": [],
                "force": [],
                "current": [],
                "tactile_raw": [],
                "tactile": [],
                "action": [],
            }
            self.save_event.set()

    def stop(self):
        with self.lock:
            self.save_event.clear()
            data = self.data
            save_path = self.save_path
            self.data = None
            self.save_path = None

        if data is None or save_path is None:
            return

        for name, values in data.items():
            np.save(os.path.join(save_path, f"{name}.npy"), np.array(values))

    def end(self):
        self.exit_event.set()
        self.control_thread.join()
        try:
            self.ser.close()
        except Exception:
            pass
        if self.save_event.is_set():
            self.stop()

    def move(self, action: np.ndarray):
        assert action.shape == (ACTION_DOF,)
        with self.lock:
            self.action = action.copy()

    def get_data(self):
        with self.lock:
            qpos = None if self.latest_qpos is None else self.latest_qpos.copy()
            tactile_raw = None if self.latest_tactile_raw is None else self.latest_tactile_raw.copy()
            tactile = None if self.latest_tactile is None else self.latest_tactile.copy()
            cur_time = self.latest_time if self.latest_time is not None else time.time()
        return {
            "qpos": qpos,
            "position": None,
            "tactile_raw": tactile_raw,
            "tactile": tactile,
            "time": cur_time
        }

    def is_error(self):
        return self.error_event.is_set()
