import time
import os
from threading import Thread, Event, Lock
from typing import Optional, List

import numpy as np
from pymodbus.client.sync import ModbusTcpClient  # pip3 install pymodbus==2.5.3

# RH56F1 Modbus TCP register map (from inspire-hand-ros2)
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
TACTILE_REG_COUNT = 34

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


def _to_raw_targets(action: np.ndarray) -> List[int]:
    # action: [0..1000] per DOF, open=1000, close=0
    if action.shape != (ACTION_DOF,):
        raise ValueError("action must be shape (6,)")
    limits = [1740, 1740, 1740, 1740, 1350, 1800]
    out = []
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
    if raw_regs is None or raw_regs.size < TACTILE_REG_COUNT:
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


class InspireF1ControllerIP:
    def __init__(self, ip: str, port: int, tactile: bool = False):
        self.ip = ip
        self.port = port
        self.tactile = tactile

        self.save_event = Event()
        self.exit_event = Event()
        self.connection_event = Event()
        self.lock = Lock()

        self.action = np.zeros(ACTION_DOF, dtype=np.float64) + 1000
        self.latest_qpos = None
        self.latest_force = None
        self.latest_current = None
        self.latest_tactile = None
        self.latest_tactile_raw = None
        self.latest_time = None

        self.save_path = None
        self.data = None

        self.client = ModbusTcpClient(self.ip, self.port)
        self.client.connect()

        self._init_device()

        self.thread = Thread(target=self.control_loop, daemon=True)
        self.thread.start()
        self.connection_event.set()

    def _init_device(self):
        try:
            self.write_registers(ADDR_CLEAR_ERROR, [1])
        except Exception:
            pass
        self.write_registers(ADDR_MOTION_MODE, [0] * ACTION_DOF)
        self.write_registers(ADDR_SET_SPEED, [1000] * ACTION_DOF)
        self.write_registers(ADDR_SET_FORCE, [300] * ACTION_DOF)

    def write_registers(self, address: int, values: List[int]) -> bool:
        response = self.client.write_registers(address, values)
        return response.isError() is False

    def read_registers(self, address: int, count: int) -> Optional[List[int]]:
        response = self.client.read_holding_registers(address, count)
        if response.isError():
            return None
        return response.registers

    def control_loop(self):
        self.fps = 30
        while not self.exit_event.is_set():
            start_time = time.time()
            with self.lock:
                action = self.action.copy()

            try:
                targets = _to_raw_targets(action)
                self.write_registers(ADDR_SET_ANGLE, targets)

                raw_angle = self.read_registers(ADDR_ACTUAL_ANGLE, ACTION_DOF)
                raw_force = self.read_registers(ADDR_ACTUAL_FORCE, ACTION_DOF)
                raw_current = self.read_registers(ADDR_ACTUATOR_CURRENT, ACTION_DOF)
                raw_tactile = self.read_registers(ADDR_TACTILE, TACTILE_REG_COUNT) if self.tactile else None

                if raw_angle is not None:
                    self.latest_qpos = _raw_to_qpos(np.asarray(raw_angle, dtype=np.float64))
                if raw_force is not None:
                    self.latest_force = np.asarray(raw_force, dtype=np.float64)
                if raw_current is not None:
                    self.latest_current = np.asarray(raw_current, dtype=np.float64)
                if raw_tactile is not None:
                    raw_arr = np.asarray(raw_tactile, dtype=np.uint16)
                    self.latest_tactile_raw = raw_arr
                    self.latest_tactile = _decode_tactile(raw_arr.astype(np.float64))
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
                pass

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
        self.thread.join()
        try:
            self.client.close()
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
        return False
