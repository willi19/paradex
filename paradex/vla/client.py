"""Shared VLA-policy client helpers used by both:
  - src/stream/vla/viewer.py            (GUI viewer + Query/Execute buttons)
  - vla_test/robot_server/robot_client.py (terminal REPL: q/e/x)

Conventions (must match training data + the server's expected request schema):
  - All angles in radians.
  - Hand qpos order: [thumb_yaw, thumb_pitch, index, middle, ring, pinky].
    Raw register order from Inspire SDK is the REVERSE.
  - `state.eef_position` / `state.eef_rotation` are in the `link_base` frame
    of the xArm6 (NOT the xArm SDK's internal cartesian frame), computed by
    FK on the same URDF used during training.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

# ── Protocol constants (locked by training data) ────────────────────────────
POS_SCALE = 0.05
ROT_SCALE = 0.5
ACTION_CHUNK_LENGTH = 16
# Per-joint max radian for Inspire, in qpos order [thumb_yaw..pinky].
INSPIRE_LIMITS_RAD = np.array([1.15, 0.55, 1.6, 1.6, 1.6, 1.6])
ARM_JOINT_NAMES = [f"joint{i + 1}" for i in range(6)]

# xarm_inspire URDF: link6 -> wrist via fixed `arm_to_hand` joint
# (rsc/robot/xarm_inspire.urdf and vla_test/.../xarm_inspire.urdf agree).
#   rpy = (π, 0, 0)   xyz = (0, 0, 0.035)
ARM_TO_HAND_OFFSET = np.eye(4)
ARM_TO_HAND_OFFSET[:3, :3] = R.from_euler('xyz', [np.pi, 0, 0]).as_matrix()
ARM_TO_HAND_OFFSET[:3, 3] = [0.0, 0.0, 0.035]


# ── Robot state I/O ──────────────────────────────────────────────────────────

def read_arm_qpos_rad(arm) -> np.ndarray:
    """xArm joint1..6 (rad). Slice [:6] in case the SDK returns 7 (xArm7 pad)."""
    q = np.asarray(arm.get_data()['qpos'], dtype=np.float64)
    return q[:6]


def _inspire_read_raw(hand) -> np.ndarray:
    """Inspire raw `angleAct` registers, 0..1000.
    Order: [little, ring, middle, index, thumb_pitch, thumb_yaw]."""
    if hasattr(hand, "get_qpos"):           # InspireControllerIP (Modbus/TCP)
        return np.asarray(hand.get_qpos(), dtype=np.int32)
    return np.asarray(hand.get_data()['joint_value'], dtype=np.int32)  # USB


def read_hand_qpos_rad(hand) -> np.ndarray:
    """Inspire qpos [thumb_yaw, thumb_pitch, index, middle, ring, pinky] (rad).
    register=0 → fully closed (max rad), register=1000 → fully open (0 rad).
    Matches paradex.robot.inspire.parse_inspire and src/process/teleop_real/
    unified.py:inspire_register_to_qpos — the convention VLA was trained on."""
    raw = _inspire_read_raw(hand)
    qpos_order_register = raw[::-1]
    return INSPIRE_LIMITS_RAD * (1.0 - qpos_order_register / 1000.0)


def hand_qpos_rad_to_raw_register(hand_qpos_rad) -> np.ndarray:
    """Inverse of read_hand_qpos_rad: qpos [thumb_yaw..pinky] (rad)
    → Inspire raw register order [little..thumb_yaw], int32, clipped to [0, 1000].
    qpos=limit (closed) → register=0; qpos=0 (open) → register=1000."""
    q = np.asarray(hand_qpos_rad, dtype=np.float64)
    raw_qpos = np.clip((1.0 - q / INSPIRE_LIMITS_RAD) * 1000.0, 0, 1000).astype(np.int32)
    return raw_qpos[::-1]


def hand_raw_register_to_qpos_rad(raw_register) -> np.ndarray:
    """Inverse of hand_qpos_rad_to_raw_register: raw register order
    [little..thumb_yaw], 0..1000 → qpos [thumb_yaw..pinky] (rad).
    register=0 → fully closed (max rad); register=1000 → open (0 rad)."""
    reg = np.clip(np.asarray(raw_register, dtype=np.float64), 0, 1000)
    qpos_order_register = reg[::-1]
    return INSPIRE_LIMITS_RAD * (1.0 - qpos_order_register / 1000.0)


# ── FK (link6 in link_base) ──────────────────────────────────────────────────

def fk_link6(urdf_or_rm, arm_joints_rad) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (pos[3] meters, rotvec[3] radians) of link6 in link_base.

    Accepts either a `yourdfpy.URDF` directly or a `paradex.visualization.robot
    .RobotModule` (which wraps one)."""
    urdf = urdf_or_rm.urdf if hasattr(urdf_or_rm, "urdf") else urdf_or_rm
    cfg = {ARM_JOINT_NAMES[i]: float(arm_joints_rad[i]) for i in range(6)}
    urdf.update_cfg(cfg)
    T = urdf.get_transform("link6", "link_base")
    pos = T[:3, 3].copy()
    rotvec = R.from_matrix(T[:3, :3]).as_rotvec()
    return pos, rotvec


# ── Image prep ───────────────────────────────────────────────────────────────

def prepare_frame_for_vla(bgr, target_hw=(480, 640)):
    """BGR ndarray (any size) → RGB (H, W, 3) uint8 for `video.front`.
    The stream tiles are ~1/8 of the sensor (192x256), so this upscales."""
    import cv2
    h, w = target_hw
    resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


# ── Action chunk decoding ───────────────────────────────────────────────────

def _infer_action_mode(chunk):
    """Heuristic for policy servers that do not declare their action schema."""
    first_pos = np.asarray(chunk[0, 0:3], dtype=np.float64)
    # Absolute xArm workspace positions are typically around 0.3-0.8m in x/z.
    # Normalized deltas are small unitless values around 0.01-0.1.
    return "delta" if np.linalg.norm(first_pos) < 0.20 else "absolute"


def decode_action_chunk(chunk, cur_pos, cur_rotvec, action_mode="auto"):
    """Action chunk (16, 12) → list of 16 absolute targets.

    action_mode:
      - "absolute": [abs_pos_m(3), abs_rotvec_rad(3), hand_qpos_rad(6)].
        This matches `src/process/teleop_real/unified.py`.
      - "delta": [delta_pos(3), delta_rotvec(3), hand_qpos_rad(6)] with
        POS_SCALE / ROT_SCALE. Kept for older normalized-delta servers.
      - "auto": infer from the first position vector magnitude.
    """
    if action_mode == "auto":
        action_mode = _infer_action_mode(chunk)
    if action_mode not in {"absolute", "delta"}:
        raise ValueError(f"unknown action_mode: {action_mode}")

    targets = []
    cur_pos = np.asarray(cur_pos, dtype=np.float64).copy()
    cur_rotvec = np.asarray(cur_rotvec, dtype=np.float64).copy()
    hand_chunk = np.asarray(chunk[:ACTION_CHUNK_LENGTH, 6:12], dtype=np.float64)
    if np.any(hand_chunk < 0.0) or np.any(hand_chunk > INSPIRE_LIMITS_RAD):
        print("[vla.client] clipping hand qpos to Inspire limits: "
              f"min={np.round(np.nanmin(hand_chunk, axis=0), 3).tolist()} "
              f"max={np.round(np.nanmax(hand_chunk, axis=0), 3).tolist()} "
              f"limits={np.round(INSPIRE_LIMITS_RAD, 3).tolist()}")
    for k in range(ACTION_CHUNK_LENGTH):
        if action_mode == "absolute":
            cur_pos = np.asarray(chunk[k, 0:3], dtype=np.float64).copy()
            cur_rotvec = np.asarray(chunk[k, 3:6], dtype=np.float64).copy()
        else:
            delta_pos_m = chunk[k, 0:3] * POS_SCALE
            delta_rot_rad = chunk[k, 3:6] * ROT_SCALE
            cur_pos = cur_pos + delta_pos_m
            cur_R = R.from_rotvec(delta_rot_rad) * R.from_rotvec(cur_rotvec)
            cur_rotvec = cur_R.as_rotvec()
        target_hand_rad = np.clip(
            np.asarray(chunk[k, 6:12], dtype=np.float64),
            0.0,
            INSPIRE_LIMITS_RAD,
        )
        targets.append({
            "target_pos_m":         cur_pos.copy(),
            "target_rotvec_rad":    cur_rotvec.copy(),
            "target_hand_rad_qpos": target_hand_rad,
            "target_hand_raw_reg":  hand_qpos_rad_to_raw_register(target_hand_rad),
        })
    return targets


def integrate_chunk(chunk, cur_pos, cur_rotvec, action_mode="auto"):
    """Backward-compatible wrapper for callers using the old helper name."""
    return decode_action_chunk(chunk, cur_pos, cur_rotvec, action_mode)


# ── Cartesian execution ──────────────────────────────────────────────────────

def target_to_arm_homo(target, link6_to_sdk_tcp=None) -> np.ndarray:
    """Build the 4x4 homogeneous matrix that paradex's xarm_controller wants:
    base-frame, METERS for translation. xarm_controller does
    `homo2aa` internally → set_servo_cartesian_aa (m→mm scaling, rotvec rad)."""
    homo = np.eye(4)
    homo[:3, :3] = R.from_rotvec(target["target_rotvec_rad"]).as_matrix()
    homo[:3,  3] = target["target_pos_m"]
    if link6_to_sdk_tcp is not None:
        homo = homo @ np.asarray(link6_to_sdk_tcp, dtype=np.float64)
    return homo


def send_cartesian_step(arm, hand, target, link6_to_sdk_tcp=None):
    """Push one integrated target to the real robot.

    arm  : paradex XArmController  (move() accepts 4x4 homo in meters)
    hand : paradex InspireController(IP)  (move() accepts 6-int raw register)
    target: one dict from integrate_chunk()
    """
    arm.move(target_to_arm_homo(target, link6_to_sdk_tcp), is_servo=True)
    hand.move(target["target_hand_raw_reg"])


def interpolate_targets(start_pos, start_rotvec, start_hand_rad, target,
                        max_pos_step_m=0.02, max_rot_step_rad=0.15):
    """Split a target into bounded Cartesian subtargets from the current pose."""
    start_pos = np.asarray(start_pos, dtype=np.float64)
    start_rotvec = np.asarray(start_rotvec, dtype=np.float64)
    end_pos = np.asarray(target["target_pos_m"], dtype=np.float64)
    end_rotvec = np.asarray(target["target_rotvec_rad"], dtype=np.float64)
    start_hand = np.asarray(start_hand_rad, dtype=np.float64)
    end_hand = np.asarray(target["target_hand_rad_qpos"], dtype=np.float64)

    dpos = np.linalg.norm(end_pos - start_pos)
    drot_vec = (R.from_rotvec(end_rotvec) *
                R.from_rotvec(start_rotvec).inv()).as_rotvec()
    drot = np.linalg.norm(drot_vec)
    n_pos = int(np.ceil(dpos / max(max_pos_step_m, 1e-6)))
    n_rot = int(np.ceil(drot / max(max_rot_step_rad, 1e-6)))
    n = max(1, n_pos, n_rot)

    start_R = R.from_rotvec(start_rotvec)
    delta_R = R.from_rotvec(drot_vec)
    out = []
    for i in range(1, n + 1):
        a = i / n
        pos = start_pos + (end_pos - start_pos) * a
        rotvec = (R.from_rotvec(drot_vec * a) * start_R).as_rotvec()
        hand_rad = start_hand + (end_hand - start_hand) * a
        out.append({
            "target_pos_m": pos,
            "target_rotvec_rad": rotvec,
            "target_hand_rad_qpos": hand_rad,
            "target_hand_raw_reg": hand_qpos_rad_to_raw_register(hand_rad),
        })
    return out


def execute_plan(arm, hand, targets, control_hz: float = 30.0,
                 on_step=None, stop_check=None, link6_to_sdk_tcp=None,
                 start_pos=None, start_rotvec=None, start_hand_rad=None,
                 max_pos_step_m=0.02, max_rot_step_rad=0.15):
    """Iterate every target at `control_hz`. Sleeps to maintain rate.

    on_step(idx, target) : optional callback after each send (for logging/viz)
    stop_check()         : optional bool callable — if True, abort mid-plan
    Returns the number of steps actually sent."""
    dt = 1.0 / control_hz
    sent = 0
    cur_pos = np.asarray(start_pos if start_pos is not None else targets[0]["target_pos_m"],
                         dtype=np.float64).copy()
    cur_rotvec = np.asarray(start_rotvec if start_rotvec is not None else targets[0]["target_rotvec_rad"],
                            dtype=np.float64).copy()
    cur_hand = np.asarray(start_hand_rad if start_hand_rad is not None else targets[0]["target_hand_rad_qpos"],
                          dtype=np.float64).copy()
    for idx, target in enumerate(targets):
        subtargets = interpolate_targets(
            cur_pos, cur_rotvec, cur_hand, target,
            max_pos_step_m=max_pos_step_m,
            max_rot_step_rad=max_rot_step_rad,
        )
        for sub_idx, subtarget in enumerate(subtargets):
            if stop_check is not None and stop_check():
                return sent
            loop_start = time.time()
            send_cartesian_step(arm, hand, subtarget, link6_to_sdk_tcp)
            if on_step is not None:
                try:
                    on_step(idx, subtarget)
                except Exception as e:
                    print(f"[vla.client] on_step({idx}.{sub_idx}) raised: {e}")
            sent += 1
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
        cur_pos = np.asarray(target["target_pos_m"], dtype=np.float64).copy()
        cur_rotvec = np.asarray(target["target_rotvec_rad"], dtype=np.float64).copy()
        cur_hand = np.asarray(target["target_hand_rad_qpos"], dtype=np.float64).copy()
    return sent


# ── Safety check (used by both viewer Execute button + robot_client) ────────

def check_plan_safe(targets, max_step_pos_m: float = 0.10,
                    max_step_rot_rad: float = 0.6,
                    start_pos=None, start_rotvec=None,
                    max_start_pos_m: float = 0.12,
                    max_start_rot_rad: float = 0.8,
                    workspace_min=None,
                    workspace_max=None,
                    hand_min_rad=None,
                    hand_max_rad=None) -> Tuple[bool, str]:
    """Reject plans with absurd per-step deltas (server bug, bad init, etc.).

    Returns (ok, reason). Default thresholds: 10 cm / step in position,
    ~34° / step in rotation. Step rate is ~30 Hz, so these are already
    very generous for a real manipulator."""
    if not targets:
        return False, "empty plan"

    prev_pos = np.asarray(targets[0]["target_pos_m"], dtype=np.float64)
    prev_rotvec = np.asarray(targets[0]["target_rotvec_rad"], dtype=np.float64)

    if start_pos is not None:
        start_pos = np.asarray(start_pos, dtype=np.float64)
        dp0 = np.linalg.norm(prev_pos - start_pos)
        if dp0 > max_start_pos_m:
            return False, f"start→step0 |dpos|={dp0:.3f}m > {max_start_pos_m}m"

    if start_rotvec is not None:
        start_rotvec = np.asarray(start_rotvec, dtype=np.float64)
        R_start = R.from_rotvec(start_rotvec)
        R_first = R.from_rotvec(prev_rotvec)
        dr0 = np.linalg.norm((R_first * R_start.inv()).as_rotvec())
        if dr0 > max_start_rot_rad:
            return False, f"start→step0 |drot|={dr0:.3f}rad > {max_start_rot_rad}rad"

    if workspace_min is not None and workspace_max is not None:
        wmin = np.asarray(workspace_min, dtype=np.float64)
        wmax = np.asarray(workspace_max, dtype=np.float64)
        for i, t in enumerate(targets):
            pos = np.asarray(t["target_pos_m"], dtype=np.float64)
            if np.any(pos < wmin) or np.any(pos > wmax):
                return False, (
                    f"step {i}: pos={np.round(pos, 3).tolist()} outside "
                    f"[{np.round(wmin, 3).tolist()}, {np.round(wmax, 3).tolist()}]"
                )

    if hand_min_rad is not None and hand_max_rad is not None:
        hmin = np.asarray(hand_min_rad, dtype=np.float64)
        hmax = np.asarray(hand_max_rad, dtype=np.float64)
        for i, t in enumerate(targets):
            h = np.asarray(t["target_hand_rad_qpos"], dtype=np.float64)
            below = h < hmin
            above = h > hmax
            if np.any(below) or np.any(above):
                return False, (
                    f"step {i}: hand qpos outside limits "
                    f"value={np.round(h, 3).tolist()} "
                    f"min={np.round(hmin, 3).tolist()} "
                    f"max={np.round(hmax, 3).tolist()}"
                )

    for i, t in enumerate(targets[1:], start=1):
        dp = np.linalg.norm(t["target_pos_m"] - prev_pos)
        if dp > max_step_pos_m:
            return False, f"step {i}: |dpos|={dp:.3f}m > {max_step_pos_m}m"
        # angular delta via composed rotation
        R_prev = R.from_rotvec(prev_rotvec)
        R_curr = R.from_rotvec(t["target_rotvec_rad"])
        R_delta = R_curr * R_prev.inv()
        dr = np.linalg.norm(R_delta.as_rotvec())
        if dr > max_step_rot_rad:
            return False, f"step {i}: |drot|={dr:.3f}rad > {max_step_rot_rad}rad"
        prev_pos = t["target_pos_m"]
        prev_rotvec = t["target_rotvec_rad"]
    return True, "ok"


# ── ZMQ msgpack client (lazy imports — keep this module zero-dep at top) ────

def _as_chunk_2d(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"expected action chunk with 2 or 3 dims, got {arr.shape}")
    return arr


def _summarize_action_array(name, arr):
    arr = np.asarray(arr)
    flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim >= 1 else arr.reshape(1, 1)
    if arr.size and np.issubdtype(arr.dtype, np.number):
        print(f"[vla.gr00t raw] {name}: shape={arr.shape} dtype={arr.dtype} "
              f"min={np.round(np.nanmin(flat, axis=0), 4).tolist()} "
              f"max={np.round(np.nanmax(flat, axis=0), 4).tolist()} "
              f"first={np.round(flat[0], 4).tolist()}")
    else:
        print(f"[vla.gr00t raw] {name}: shape={arr.shape} dtype={arr.dtype}")


class _ZmqMsgpackClient:
    """Shared REQ/msgpack transport with reconnect-on-failure behavior."""

    def __init__(self, server_url: str, send_timeout_s: float = 5.0,
                 recv_timeout_s: float = 30.0):
        import zmq, msgpack, msgpack_numpy
        from threading import Lock
        msgpack_numpy.patch()
        self._zmq = zmq
        self._msgpack = msgpack
        self._mp_encode = msgpack_numpy.encode
        self.server_url = server_url
        self.ctx = zmq.Context()
        self._send_timeout_s = send_timeout_s
        self._recv_timeout_s = recv_timeout_s
        self._lock = Lock()
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(self._zmq.REQ)
        self.sock.setsockopt(self._zmq.SNDTIMEO, int(self._send_timeout_s * 1000))
        self.sock.setsockopt(self._zmq.RCVTIMEO, int(self._recv_timeout_s * 1000))
        self.sock.setsockopt(self._zmq.LINGER, 0)
        self.sock.connect(self.server_url)

    def reconnect(self):
        try:
            self.sock.close(linger=0)
        except Exception:
            pass
        self._connect()

    def _roundtrip(self, request):
        payload = self._msgpack.packb(
            request, default=self._mp_encode, use_bin_type=True)
        with self._lock:
            t0 = time.time()
            try:
                self.sock.send(payload)
            except self._zmq.ZMQError:
                # Usually EFSM after a previous timeout/interrupt. Nothing was
                # sent successfully, so it is safe to rebuild and retry once.
                self.reconnect()
                self.sock.send(payload)
            try:
                reply = self._msgpack.unpackb(self.sock.recv(), raw=False)
            except self._zmq.ZMQError:
                self.reconnect()
                raise
            return reply, (time.time() - t0) * 1000

    def close(self):
        try:
            self.sock.close()
            self.ctx.term()
        except Exception:
            pass


def _first_frame(frame_rgb):
    if isinstance(frame_rgb, dict):
        if not frame_rgb:
            raise ValueError("empty camera frame dict")
        return next(iter(frame_rgb.values()))
    return frame_rgb


class LegacyPolicyClient(_ZmqMsgpackClient):
    """Client for the original paradex flat-dict policy server."""
    default_action_mode = "auto"

    def query(self, obs_for_policy: dict, frame_rgb: np.ndarray, task_text: str):
        """Returns (chunk_arr[T, 12], rtt_ms, server_ms)."""
        frame_rgb = _first_frame(frame_rgb)
        request = {
            **{k: np.asarray(v, dtype=np.float64) for k, v in obs_for_policy.items()},
            "video.front": frame_rgb.astype(np.uint8),
            "language.instruction": task_text,
        }
        reply, rtt_ms = self._roundtrip(request)
        chunk_arr = np.concatenate([
            reply["action.eef_position"],
            reply["action.eef_rotation"],
            reply["action.dexhand_position"],
        ], axis=1)  # (T, 12)
        return chunk_arr, rtt_ms, reply.get("elapsed_ms", -1.0)


class Gr00TPolicyClient(_ZmqMsgpackClient):
    """Client for NVIDIA GR00T PolicyServer endpoint RPC.

    Local GR00T servers have used more than one action convention in practice.
    Keep the wire protocol hidden here, but make the value basis explicit so
    the viewer can stay protocol-agnostic.
    """

    def __init__(self, server_url: str, arm_basis: str = "absolute_eef",
                 hand_basis: str = "absolute_raw", fk_model=None, **kwargs):
        self.arm_basis = arm_basis
        self.hand_basis = hand_basis
        self.fk_model = fk_model
        if self.arm_basis == "absolute_eef":
            self.default_action_mode = "absolute"
        elif self.arm_basis == "delta_eef_scaled":
            self.default_action_mode = "delta"
        elif self.arm_basis in {"delta_eef_m", "joint_delta", "joint_absolute"}:
            self.default_action_mode = "absolute"
        else:
            raise ValueError(f"unknown GR00T arm_basis: {self.arm_basis}")
        if self.arm_basis in {"joint_delta", "joint_absolute"} and self.fk_model is None:
            raise ValueError(f"{self.arm_basis} requires fk_model")
        if self.hand_basis not in {
            "absolute_raw", "delta_raw", "absolute_qpos", "delta_qpos",
        }:
            raise ValueError(f"unknown GR00T hand_basis: {self.hand_basis}")
        super().__init__(server_url, **kwargs)

    def get_modality_config(self):
        reply, _ = self._roundtrip({"endpoint": "get_modality_config"})
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"Policy server error: {reply['error']}")
        return reply

    def _build_request(self, obs_for_policy, frame_rgb, task_text):
        # Observation is grouped by modality (video/state/language), and every
        # value carries a leading (B=1, T=1) axis. Verified live against the
        # GR00T PolicyServer: video=uint8 (B,T,H,W,C), state=float32 (B,T,D),
        # language=list of shape (B,T). The model expects all of cam0..cam3.
        arm = np.asarray(obs_for_policy["state.arm_joint_position"],
                         dtype=np.float32).reshape(1, 1, -1)
        # state.hand is the Inspire RAW register (0..1000), NOT radians — that is
        # the value space the policy was trained on. obs carries hand in rad, so
        # convert back to raw register here.
        hand_reg = hand_qpos_rad_to_raw_register(
            obs_for_policy["state.dexhand_position"])
        hand = np.asarray(hand_reg, dtype=np.float32).reshape(1, 1, -1)

        frames = frame_rgb if isinstance(frame_rgb, dict) else {"cam0": frame_rgb}
        video = {}
        for key, image in frames.items():
            img = np.asarray(image, dtype=np.uint8)
            if img.ndim == 3:                       # (H,W,C) -> (B,T,H,W,C)
                img = img[None, None]
            video[key] = img

        observation = {
            "video": video,
            "state": {"arm": arm, "hand": hand},
            "language": {"annotation.human.task_description": [[task_text]]},
        }
        return {
            "endpoint": "get_action",
            "data": {"observation": observation, "options": None},
        }

    def _print_arm_basis_candidates(self, arm, obs_for_policy):
        start_pos = np.asarray(obs_for_policy["state.eef_position"], dtype=np.float64)
        start_rot = np.asarray(obs_for_policy["state.eef_rotation"], dtype=np.float64)
        a0 = np.asarray(arm[0, :6], dtype=np.float64)

        abs_pos = a0[:3]
        abs_rot = a0[3:6]
        abs_dp_mm = np.linalg.norm(abs_pos - start_pos) * 1000.0
        abs_dr_deg = np.linalg.norm(
            (R.from_rotvec(abs_rot) * R.from_rotvec(start_rot).inv()).as_rotvec()
        ) * 180.0 / np.pi

        scaled_dp_mm = np.linalg.norm(a0[:3] * POS_SCALE) * 1000.0
        scaled_dr_deg = np.linalg.norm(a0[3:6] * ROT_SCALE) * 180.0 / np.pi

        raw_dp_mm = np.linalg.norm(a0[:3]) * 1000.0
        raw_dr_deg = np.linalg.norm(a0[3:6]) * 180.0 / np.pi

        print("[vla.gr00t interp] step0 if absolute_eef: "
              f"{abs_dp_mm:.1f}mm/{abs_dr_deg:.1f}deg from current")
        print("[vla.gr00t interp] step0 if delta_eef_scaled: "
              f"{scaled_dp_mm:.1f}mm/{scaled_dr_deg:.1f}deg from current")
        print("[vla.gr00t interp] step0 if delta_eef_m: "
              f"{raw_dp_mm:.1f}mm/{raw_dr_deg:.1f}deg from current")
        if self.fk_model is not None:
            cur_joints = np.asarray(
                obs_for_policy["state.arm_joint_position"], dtype=np.float64)
            for name, joints in (
                ("joint_delta", cur_joints + a0[:6]),
                ("joint_absolute", a0[:6]),
            ):
                try:
                    j_pos, j_rot = fk_link6(self.fk_model, joints)
                    j_dp_mm = np.linalg.norm(j_pos - start_pos) * 1000.0
                    j_dr_deg = np.linalg.norm(
                        (R.from_rotvec(j_rot) *
                         R.from_rotvec(start_rot).inv()).as_rotvec()
                    ) * 180.0 / np.pi
                    print(f"[vla.gr00t interp] step0 if {name}: "
                          f"{j_dp_mm:.1f}mm/{j_dr_deg:.1f}deg from current")
                except Exception as e:
                    print(f"[vla.gr00t interp] step0 if {name}: failed ({e})")

    def _decode_arm(self, arm, obs_for_policy):
        if self.arm_basis == "absolute_eef":
            return arm
        if self.arm_basis == "delta_eef_scaled":
            return arm
        if self.arm_basis in {"joint_delta", "joint_absolute"}:
            cur_joints = np.asarray(
                obs_for_policy["state.arm_joint_position"], dtype=np.float64)
            out = []
            if self.arm_basis == "joint_delta":
                joint_chunk = cur_joints[None, :] + np.cumsum(arm[:, :6], axis=0)
            else:
                joint_chunk = arm[:, :6]
            for joints in joint_chunk:
                pos, rotvec = fk_link6(self.fk_model, joints)
                out.append(np.concatenate([pos, rotvec]))
            return np.asarray(out, dtype=np.float64)

        cur_pos = np.asarray(obs_for_policy["state.eef_position"],
                             dtype=np.float64).copy()
        cur_rotvec = np.asarray(obs_for_policy["state.eef_rotation"],
                                dtype=np.float64).copy()
        out = []
        for row in arm:
            cur_pos = cur_pos + np.asarray(row[:3], dtype=np.float64)
            cur_rotvec = (
                R.from_rotvec(row[3:6]) * R.from_rotvec(cur_rotvec)
            ).as_rotvec()
            out.append(np.concatenate([cur_pos.copy(), cur_rotvec.copy()]))
        return np.asarray(out, dtype=np.float64)

    def _decode_hand(self, hand, obs_for_policy):
        cur_qpos = np.asarray(obs_for_policy["state.dexhand_position"],
                              dtype=np.float64)
        if self.hand_basis == "absolute_qpos":
            return np.clip(hand, 0.0, INSPIRE_LIMITS_RAD)
        if self.hand_basis == "delta_qpos":
            return np.clip(cur_qpos[None, :] + np.cumsum(hand, axis=0),
                           0.0, INSPIRE_LIMITS_RAD)

        cur_raw = hand_qpos_rad_to_raw_register(cur_qpos).astype(np.float64)
        if self.hand_basis == "absolute_raw":
            raw = hand
        else:
            raw = cur_raw[None, :] + np.cumsum(hand, axis=0)
        raw = np.clip(raw, 0, 1000)
        return np.stack([
            hand_raw_register_to_qpos_rad(raw[k]) for k in range(raw.shape[0])
        ])

    def _decode_reply(self, reply, obs_for_policy):
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"Policy server error: {reply['error']}")
        action = reply[0] if isinstance(reply, (list, tuple)) else reply
        if not isinstance(action, dict):
            raise ValueError(f"unexpected GR00T reply type: {type(action)}")
        print(f"[vla.gr00t raw] action keys = {list(action.keys())}")
        if "arm" in action:
            _summarize_action_array("arm", action["arm"])
        if "hand" in action:
            _summarize_action_array("hand", action["hand"])
        arm = _as_chunk_2d(action["arm"])
        hand = _as_chunk_2d(action["hand"])
        if arm.shape[0] != hand.shape[0]:
            raise ValueError(f"arm/hand horizon mismatch: {arm.shape} vs {hand.shape}")
        print(f"[vla.gr00t interp] using arm_basis={self.arm_basis} "
              f"hand_basis={self.hand_basis} "
              f"viewer_action_mode={self.default_action_mode}")
        self._print_arm_basis_candidates(arm, obs_for_policy)
        arm = self._decode_arm(arm, obs_for_policy)
        hand_rad = self._decode_hand(hand, obs_for_policy)
        chunk_arr = np.concatenate([arm, hand_rad], axis=1)  # (T,12)
        return chunk_arr, -1.0

    def query(self, obs_for_policy: dict, frame_rgb: np.ndarray, task_text: str):
        """Returns (chunk_arr[T, 12], rtt_ms, server_ms)."""
        request = self._build_request(obs_for_policy, frame_rgb, task_text)
        reply, rtt_ms = self._roundtrip(request)
        chunk_arr, server_ms = self._decode_reply(reply, obs_for_policy)
        return chunk_arr, rtt_ms, server_ms


def make_policy_client(protocol: str, server_url: str, **kwargs):
    if protocol == "legacy":
        return LegacyPolicyClient(server_url, **kwargs)
    if protocol == "gr00t":
        return Gr00TPolicyClient(server_url, **kwargs)
    raise ValueError(f"unknown policy protocol: {protocol}")


# Backward-compatible name for older imports.
PolicyClient = LegacyPolicyClient


# ── Camera frame source (DataCollector wrapper) ──────────────────────────────

class CameraFrameSource:
    """Background thread that decodes the latest JPEG from one (or all)
    capture-PC streams into a serial→BGR dict. Used by both viewer and
    robot_client. Caller is responsible for bringing up the stream daemon."""

    def __init__(self, dc, decode_serials=None, poll_hz: float = 30.0):
        """
        dc : paradex DataCollector (already .start()-ed)
        decode_serials : iterable of serials to decode, or None for ALL
        poll_hz : how often the thread re-checks dc.get_data()
        """
        import threading
        self._dc = dc
        self._decode_serials = set(decode_serials) if decode_serials else None
        self._period = 1.0 / poll_hz
        self._lock = threading.Lock()
        self._frames = {}            # serial -> BGR ndarray
        self._last_fid = {}          # serial -> last decoded frame_id
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        import cv2
        while not self._stop.is_set():
            tic = time.time()
            data = self._dc.get_data()
            for serial, item in data.items():
                if self._decode_serials is not None and serial not in self._decode_serials:
                    continue
                if item.get('type') != 'image':
                    continue
                img_bytes = item.get('data')
                if not img_bytes:
                    continue
                fid = item.get('frame_id', -1)
                if fid <= self._last_fid.get(serial, -1):
                    continue
                arr = np.frombuffer(img_bytes, np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                self._last_fid[serial] = fid
                with self._lock:
                    self._frames[serial] = bgr
            elapsed = time.time() - tic
            if elapsed < self._period:
                time.sleep(self._period - elapsed)

    def latest(self, serial: str):
        """Returns latest BGR for `serial`, or None."""
        with self._lock:
            bgr = self._frames.get(serial)
        return None if bgr is None else bgr.copy()

    def latest_all(self) -> dict:
        with self._lock:
            return {k: v.copy() for k, v in self._frames.items()}

    def available_serials(self):
        with self._lock:
            return list(self._frames.keys())

    def stop(self):
        self._stop.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
