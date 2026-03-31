import argparse
import ctypes
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import viser
from viser.extras import ViserUrdf


HAND_DOF = 16
KIN_SENSOR_NUM = 4
KIN_SENSOR_DOF = 3
TACTILE_NUM = 60

SHM_MSG_KEY = 0x2969
PERMISSIONS = 0o666

# Encoder range per actuator index J00..J15
JOINT_ENCODER_LIMITS = [
    (0, 4096),      # J00
    (-4096, 4096),  # J01
    (0, 4096),      # J02
    (0, 4096),      # J03
    (-680, 680),    # J04
    (0, 4096),      # J05
    (0, 4096),      # J06
    (0, 4096),      # J07
    (-680, 680),    # J08
    (0, 4096),      # J09
    (0, 4096),      # J10
    (0, 4096),      # J11
    (-680, 680),    # J12
    (0, 4096),      # J13
    (0, 4096),      # J14
    (0, 4096),      # J15
]

# URDF revolute joint limits [rad] mapped to J00..J15
JOINT_URDF_LIMITS = [
    (0.0, 1.5708),        # thumb_joint_0
    (-1.5708, 1.5708),    # thumb_joint_1
    (0.0, 1.5708),        # thumb_joint_2
    (0.0, 1.5708),        # thumb_joint_3
    (-0.261799, 0.261799),  # index_joint_0
    (0.0, 1.5708),        # index_joint_1
    (0.0, 1.5708),        # index_joint_2
    (0.0, 1.5708),        # index_joint_3
    (-0.261799, 0.261799),  # middle_joint_0
    (0.0, 1.5708),        # middle_joint_1
    (0.0, 1.5708),        # middle_joint_2
    (0.0, 1.5708),        # middle_joint_3
    (-0.261799, 0.261799),  # ring_joint_0
    (0.0, 1.5708),        # ring_joint_1
    (0.0, 1.5708),        # ring_joint_2
    (0.0, 1.5708),        # ring_joint_3
]

# J00..J15 to URDF joint name mapping
JOINT_NAME_MAP = [
    "thumb_joint_0",
    "thumb_joint_1",
    "thumb_joint_2",
    "thumb_joint_3",
    "index_joint_0",
    "index_joint_1",
    "index_joint_2",
    "index_joint_3",
    "middle_joint_0",
    "middle_joint_1",
    "middle_joint_2",
    "middle_joint_3",
    "ring_joint_0",
    "ring_joint_1",
    "ring_joint_2",
    "ring_joint_3",
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

        self.shm_ptr = None
        self.msgs = None

    def connect(self):
        shm_id = self.libc.shmget(SHM_MSG_KEY, ctypes.sizeof(SHMmsgs), PERMISSIONS)
        if shm_id < 0:
            err = ctypes.get_errno()
            raise OSError(err, f"shmget failed (key=0x{SHM_MSG_KEY:x})")

        addr = self.libc.shmat(shm_id, None, 0)
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

    def read_jpos(self):
        return [int(v) for v in self.msgs.contents.j_pos]


def map_encoder_to_rad(value: float, enc_min: float, enc_max: float, rad_min: float, rad_max: float) -> float:
    if enc_max == enc_min:
        return rad_min
    t = (value - enc_min) / (enc_max - enc_min)
    t = max(0.0, min(1.0, t))
    return rad_min + t * (rad_max - rad_min)


def build_qpos_from_jpos(j_pos: List[int], urdf_joint_names: Tuple[str, ...]) -> np.ndarray:
    jname_to_idx = {name: i for i, name in enumerate(urdf_joint_names)}
    qpos = np.zeros(len(urdf_joint_names), dtype=np.float64)
    for j_idx, j_name in enumerate(JOINT_NAME_MAP):
        if j_name not in jname_to_idx:
            continue
        enc_min, enc_max = JOINT_ENCODER_LIMITS[j_idx]
        rad_min, rad_max = JOINT_URDF_LIMITS[j_idx]
        qpos[jname_to_idx[j_name]] = map_encoder_to_rad(
            j_pos[j_idx], enc_min, enc_max, rad_min, rad_max
        )
    return qpos


def main():
    parser = argparse.ArgumentParser(description="Live KISTAR URDF viewer with SHM qpos stream")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path(__file__).resolve().parent / "Ver2_kistar_basic_JS.urdf",
        help="URDF file path",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--hz", type=float, default=60.0, help="Update frequency")
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Optional auto-stop duration for quick test",
    )
    args = parser.parse_args()

    if not args.urdf.is_file():
        raise FileNotFoundError(f"URDF not found: {args.urdf}")

    server = viser.ViserServer(host=args.host, port=args.port)
    urdf_vis = ViserUrdf(server, args.urdf, root_node_name="/kistar")
    urdf_joint_names = urdf_vis.get_actuated_joint_names()
    print(f"[viser] serving on http://{args.host}:{args.port}")
    print(f"[viser] actuated joints ({len(urdf_joint_names)}): {urdf_joint_names}")

    shm = SharedMemoryClient()
    shm.connect()

    period = 1.0 / max(args.hz, 1e-3)
    t0 = time.time()
    try:
        while True:
            j_pos = shm.read_jpos()
            qpos = build_qpos_from_jpos(j_pos, urdf_joint_names)
            urdf_vis.update_cfg(qpos)

            if args.duration_sec is not None and (time.time() - t0) >= args.duration_sec:
                break
            time.sleep(period)
    finally:
        shm.close()


if __name__ == "__main__":
    main()
