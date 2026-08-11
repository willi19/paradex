"""zerodex GR00T-N1.7 client + kinematics for paradex xArm6 + Inspire.

Talks to the finetuned zerodex checkpoint served on vclabserver2
(`run_gr00t_server.py --model_path .../zerodex_fk_v1_best/checkpoint-14000`,
tcp://147.46.15.24:8901).

This is a SEPARATE path from paradex.vla.client (LegacyPolicyClient /
Gr00TPolicyClient): the zerodex server has a different obs/action contract
(rot6d eef, 2 cameras, absolute eef_target + raw hand_cmd) that does NOT match
those clients. Everything here is byte-identical to the training pipeline
(build_zerodex_fk.py) so that what we send/decode is exactly what the checkpoint
saw. Verified: FK@T_tool + column-rot6d reproduces the dataset eef_pose to 0.0mm.

GROUND TRUTH (checkpoint-14000, dataset zerodex_fk_train)
---------------------------------------------------------
INPUT observation (B=1, T=1):
  video.main       uint8   (1,1,480,640,3) RGB   cam 26053260 (FOCUS overview)
  video.secondary  uint8   (1,1,480,640,3) RGB   cam 25305465 (wide view)
  state.arm_joint  float32 (1,1,6)   xArm6 joints [rad]
  state.hand_joint float32 (1,1,6)   Inspire OBSERVED angleAct, RAW 0..1000
                                      order [little,ring,middle,index,thumb_bend,thumb_rot]
  state.eef_pose   float32 (1,1,9)   FK(arm_joint) @ T_TOOL -> [xyz(m), rot6d]
                                      rot6d = first two COLUMNS of R
  language.annotation.human.action.task_description  [[str]]

OUTPUT action (use_relative_action=true; server returns ABSOLUTE):
  eef_target  float32 (16,9)  ABSOLUTE TCP pose [xyz(m), rot6d]. Feed to IK / send
                              cartesian; do NOT add eef_pose back.
  hand_cmd    float32 (16,6)  ABSOLUTE Inspire command, RAW 0..1000, same order.

Only 4 prompts are in-distribution (see PROMPTS).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

# ── in-distribution prompts (meta/tasks.jsonl) ──────────────────────────────
PROMPTS = [
    "move apple to the pink basket",
    "move banana to the pink basket",
    "move pepsi to the pink basket",
    "move potato to the pink basket",
]

# camera serials (build_zerodex_fk.py: CAMS)
CAM_MAIN = "26053260"       # FOCUS overview
CAM_SECONDARY = "25305465"  # wide opposite view

ACTION_HORIZON = 16
HAND_MIN, HAND_MAX = 0.0, 1000.0

# in-distribution workspace box for eef_pose xyz (dataset_statistics.json), meters
EEF_XYZ_MIN = np.array([0.2695, -0.3895, 0.1612])
EEF_XYZ_MAX = np.array([0.7583, 0.2420, 0.6136])


# ── xArm6 modified-DH FK (validated ~1.5mm/0.36deg) ─────────────────────────
_A2 = np.sqrt(284.5**2 + 53.5**2) / 1000.0
_T2 = -np.arctan2(284.5, 53.5)
_T3 = np.arctan2(284.5, 53.5)
_OFF = np.array([0.0, _T2, _T3, 0.0, 0.0, 0.0])
_DH = [(0.267, 0.0, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, _A2, 0.0),
       (0.3425, 0.0775, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.097, 0.076, -np.pi / 2)]

# zerodex-specific flange->action-TCP transform (T_tool_zerodex.npy, ~135.7mm Z).
# NOTE: this differs from the pnp_pepsi T_tool (~136.1mm Z) — using the wrong one
# biases every returned eef_target frame and the closed loop drifts.
T_TOOL = np.array([
    [0.9999938708624533, -0.000822140231160612, 0.0034032812060654023, 0.0011637081583017656],
    [0.0008243652334518512, 0.9999994473781423, -0.000652430358102475, -0.00029205452599082856],
    [-0.0034027429360923927, 0.0006552319059730108, 0.9999939959878061, 0.13568172110829674],
    [0.0, 0.0, 0.0, 1.0],
])
T_TOOL_INV = np.linalg.inv(T_TOOL)

ARM_LO = np.array([-6.28318, -2.059, -3.927, -6.28318, -1.69297, -6.28318])
ARM_HI = np.array([6.28318, 2.0944, 0.19198, 6.28318, 3.14159, 6.28318])


def _rz(t): c, s = np.cos(t), np.sin(t); return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])
def _tz(d): T = np.eye(4); T[2, 3] = d; return T
def _rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.]])
def _tx(a): T = np.eye(4); T[0, 3] = a; return T


def fk_flange(q):
    """4x4 base->flange (link6) for 6 joint angles (rad)."""
    T = np.eye(4)
    for i in range(6):
        d, a, al = _DH[i]
        T = T @ (_rx(al) @ _tx(a) @ _rz(q[i] + _OFF[i]) @ _tz(d))
    return T


def fk_tcp(q):
    """4x4 base->TCP (= flange @ T_TOOL); matches state.eef_pose frame."""
    return fk_flange(q) @ T_TOOL


# ── rot6d <-> matrix (GR00T XYZ_ROT6D = first two rotation COLUMNS) ──────────
def rot6d_to_R(r6):
    a1, a2 = r6[:3].astype(float), r6[3:6].astype(float)
    n = np.linalg.norm(a1)
    b1 = a1 / n if n > 1e-9 else np.array([1.0, 0, 0])
    a2p = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(a2p)
    b2 = a2p / n2 if n2 > 1e-9 else np.array([0.0, 1, 0])
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def R_to_rot6d(R):
    return np.concatenate([R[:, 0], R[:, 1]])


def pose9_to_T(p9):
    """[xyz(3), rot6d(6)] -> 4x4 homogeneous."""
    T = np.eye(4)
    T[:3, :3] = rot6d_to_R(p9[3:9])
    T[:3, 3] = p9[:3]
    return T


def eef_pose9(q6):
    """6 arm joints (rad) -> state.eef_pose [xyz(m,3), rot6d(6)] in the TCP frame.

    EXACTLY how build_zerodex_fk.py:fk_pose9 built state.eef_pose:
    T = FK(q) @ T_TOOL, then [translation, first-two-COLUMNS of R]. This is the
    anchor the server composes its RELATIVE eef_target against, so it MUST be
    produced this way (never from the xArm's reported cartesian).
    """
    T = fk_tcp(q6)
    return np.concatenate([T[:3, 3], R_to_rot6d(T[:3, :3])]).astype(np.float32)


def ik_tcp(target9, q_seed, iters=60, lam=0.08, eps=1e-5, tol=1e-5):
    """Damped least-squares IK: joint1..6 so fk_tcp(q) matches target [xyz, rot6d].

    Warm-started from q_seed (GT arm_joint on the first frame, then the previous
    solution) to stay on the correct IK branch. Returns (q, pos_err_m, rot_err_deg).
    """
    Tt = pose9_to_T(target9)
    p_d, R_d = Tt[:3, 3], Tt[:3, :3]
    q = np.array(q_seed, dtype=float).copy()
    for _ in range(iters):
        T = fk_tcp(q)
        p, R = T[:3, 3], T[:3, :3]
        e_p = p_d - p
        e_r = Rot.from_matrix(R_d @ R.T).as_rotvec()
        if np.linalg.norm(e_p) < tol and np.linalg.norm(e_r) < 1e-4:
            break
        J = np.zeros((6, 6))
        for i in range(6):
            dq = q.copy(); dq[i] += eps
            Ti = fk_tcp(dq)
            J[:3, i] = (Ti[:3, 3] - p) / eps
            J[3:, i] = Rot.from_matrix(Ti[:3, :3] @ R.T).as_rotvec() / eps
        JJt = J @ J.T + (lam ** 2) * np.eye(6)
        dqv = J.T @ np.linalg.solve(JJt, np.concatenate([e_p, e_r]))
        n = np.linalg.norm(dqv)
        if n > 0.3:
            dqv *= 0.3 / n
        q = np.clip(q + dqv, ARM_LO, ARM_HI)
    T = fk_tcp(q)
    pe = np.linalg.norm(T[:3, 3] - p_d)
    re = np.degrees(np.linalg.norm(Rot.from_matrix(R_d @ T[:3, :3].T).as_rotvec()))
    return q, pe, re


# ── ZMQ wire client (msgpack-numpy REQ/REP; no gr00t/torch needed) ──────────
class ZerodexPolicyClient:
    """Wire-verified client for the zerodex GR00T PolicyServer.

    Request : msgpack({"endpoint": <str>, "data": {...}}), numpy via msgpack_numpy.
    Response: msgpack(<result>); errors come back as {"error": "..."}.
    """

    def __init__(self, host="147.46.15.24", port=8901, timeout_ms=30000):
        import zmq
        import msgpack
        import msgpack_numpy as mnp
        self._zmq = zmq
        self._msgpack = msgpack
        self._mnp = mnp
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self.ctx = zmq.Context.instance()
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(self._zmq.REQ)
        self.sock.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        self.sock.setsockopt(self._zmq.LINGER, 0)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    def _call(self, endpoint, data=None):
        req = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        try:
            self.sock.send(self._msgpack.packb(req, default=self._mnp.encode))
            raw = self.sock.recv()
        except self._zmq.error.Again:
            # A REQ socket that timed out is stuck waiting for a reply that will
            # never come; rebuild it so the next call can send at all.
            self.sock.close(linger=0)
            self._connect()
            raise TimeoutError(f"zerodex server no reply within {self.timeout_ms} ms")
        rep = self._msgpack.unpackb(raw, object_hook=self._mnp.decode, raw=False)
        if isinstance(rep, dict) and "error" in rep:
            raise RuntimeError(f"server error: {rep['error']}")
        return rep

    def ping(self):
        return self._call("ping")

    def get_action(self, obs):
        """obs: nested/batched dict from build_observation.
        Returns (eef_target (H,9) ABSOLUTE, hand_cmd (H,6) RAW), batch dim removed."""
        rep = self._call("get_action", {"observation": obs})
        act = rep[0] if isinstance(rep, (list, tuple)) else rep
        eef = np.asarray(act["eef_target"], np.float32).reshape(-1, 9)
        hand = np.asarray(act["hand_cmd"], np.float32).reshape(-1, 6)
        return eef, hand

    def close(self):
        self.sock.close(linger=0)


# ── observation builder (eef_pose ONLY from FK@T_tool, never the xArm cartesian)
def build_observation(img_main, img_secondary, arm_qpos_rad, hand_raw, instruction):
    """
    img_main/img_secondary : (480,640,3) uint8 RGB
    arm_qpos_rad           : (6,) xArm6 joint angles, radians
    hand_raw               : (6,) Inspire OBSERVED angleAct, RAW 0..1000,
                             order [little,ring,middle,index,thumb_bend,thumb_rot]
    instruction            : one of PROMPTS
    """
    m = np.ascontiguousarray(img_main, np.uint8)
    s = np.ascontiguousarray(img_secondary, np.uint8)
    assert m.ndim == 3 and m.shape[-1] == 3, f"main image must be HxWx3, got {m.shape}"
    assert s.ndim == 3 and s.shape[-1] == 3, f"secondary image must be HxWx3, got {s.shape}"
    arm = np.asarray(arm_qpos_rad, np.float32).reshape(6)
    return {
        "video": {"main": m[None, None], "secondary": s[None, None]},
        "state": {
            "arm_joint": arm.reshape(1, 1, 6),
            "hand_joint": np.asarray(hand_raw, np.float32).reshape(1, 1, 6),
            "eef_pose": eef_pose9(arm).reshape(1, 1, 9),
        },
        "language": {"annotation.human.action.task_description": [[str(instruction)]]},
    }


# ── safety on the absolute eef_target chunk ─────────────────────────────────
def check_plan_safe(eef_chunk, eef_now9, max_step_pos_m=0.05, pad_m=0.05):
    """Reject a chunk with a large first jump or out-of-workspace targets.

    max_step_pos_m : max L2 position change between consecutive absolute targets
                     (and from the current pose to target[0]).
    pad_m          : padding added to the dataset workspace box.
    """
    pts = eef_chunk[:, :3].astype(np.float64)
    prev = np.asarray(eef_now9[:3], np.float64)
    for i, p in enumerate(pts):
        d = np.linalg.norm(p - prev)
        if d > max_step_pos_m:
            return False, f"step {i} jump {d*1000:.0f}mm > {max_step_pos_m*1000:.0f}mm"
        prev = p
    lo, hi = EEF_XYZ_MIN - pad_m, EEF_XYZ_MAX + pad_m
    if np.any(pts < lo) or np.any(pts > hi):
        bad = np.where(np.any((pts < lo) | (pts > hi), axis=1))[0]
        return False, f"targets {bad.tolist()} outside workspace box"
    return True, "ok"


# ── execute one absolute TCP target on paradex hardware ─────────────────────
def send_arm_cartesian(arm, target9, tool_frame="tcp"):
    """arm.move() a single absolute TCP target [xyz(m), rot6d].

    tool_frame='tcp' (DEFAULT, verified on this rig): the xArm's cartesian frame
        (get_position / set_servo_cartesian_aa) already equals FK@T_TOOL (TCP) —
        measured get_position == fk_tcp(q) to 0.5mm/0.07deg — so send eef_target
        UNCHANGED. Using 'flange' here sent the flange pose (~123mm off in x) and
        the closed loop drifted backwards.
    tool_frame='flange': xArm cartesian targets the flange -> send T @ T_TOOL_INV.
        (Correct only if the controller has NO tool offset; NOT the case here.)
    """
    T = pose9_to_T(target9)
    if tool_frame == "flange":
        T = T @ T_TOOL_INV
    arm.move(T, is_servo=True)


def couple_thumb_to_fingers(hand_cmd, couple_rot=False, gain=1.0):
    """Override the policy's thumb with a value coupled to the 4 fingers so the
    thumb grips when they grip and opens when they open.

    hand_cmd : (...,6) RAW 0..1000, order [little,ring,middle,index,thumb_bend,
    thumb_rot]. Fixes the stuck-thumb behavior baked into the training data
    (thumb_bend was ~pinned open). Drives thumb_bend from the mean finger
    openness; gain>1 closes the thumb more than the fingers. Optionally couples
    thumb_rot (opposition) the same way. Works on (6,) or (H,6)."""
    hv = np.asarray(hand_cmd, dtype=np.float32).copy()
    finger_open = hv[..., 0:4].mean(axis=-1)                  # 0(closed)..1000(open)
    thumb = np.clip(1000.0 - gain * (1000.0 - finger_open), 0.0, 1000.0)
    hv[..., 4] = thumb                                        # thumb_bend
    if couple_rot:
        hv[..., 5] = thumb                                    # thumb_rot / opposition
    return hv


def send_hand(hand, hand_cmd6):
    """Send one absolute Inspire command (RAW 0..1000, native order)."""
    reg = np.clip(np.asarray(hand_cmd6, np.float32), HAND_MIN, HAND_MAX).astype(np.int32)
    hand.move(reg)
