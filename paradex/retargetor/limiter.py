"""Velocity-based outlier rejection for 4x4 wrist pose commands.

VIVE trackers occasionally emit a single-frame position/orientation spike
(occlusion, IMU reset). Feeding that straight to arm.move() makes the robot
lurch. This limiter rejects any candidate whose per-step translation/rotation
exceeds max_speed*dt + margin (a physically-impossible jump for a real hand)
and tells the caller to HOLD the last accepted pose instead.

Accepted deltas are integrated onto the last *sent* pose, so a rejected spike
(or a full tracking dropout) does not offset later motion: after tracking
returns, the robot resumes moving *relatively* from where it is, rather than
teleporting to the new absolute tracker position.
"""
import numpy as np
from scipy.spatial.transform import Rotation as _R

# Slew caps: how fast the arm target is allowed to FOLLOW the hand. Lower =
# stronger limiting (smoother, but laggier on fast intentional motion).
VIVE_MAX_LINEAR_SPEED_M_S = 0.35
VIVE_MAX_ANGULAR_SPEED_DEG_S = 150.0
VIVE_POSITION_MARGIN_M = 0.003
VIVE_ROTATION_MARGIN_DEG = 1.5
VIVE_MAX_COMMAND_DT_S = 0.05     # cap dt so a long stall can't inflate the limit

# Outlier REJECT thresholds: a per-frame RAW tracker jump faster than a hand can
# physically move is a VIVE glitch -> hold in place instead of slewing toward it.
# Set well above the fastest intentional teleop motion (~1 m/s), well below a
# glitch (a 0.5 m jump in one 10 ms frame = 50 m/s).
VIVE_REJECT_LINEAR_SPEED_M_S = 2.0
VIVE_REJECT_ANGULAR_SPEED_DEG_S = 500.0


class PoseCommandLimiter:
    def __init__(self, initial_pose, timestamp,
                 max_linear=VIVE_MAX_LINEAR_SPEED_M_S,
                 max_angular_deg=VIVE_MAX_ANGULAR_SPEED_DEG_S,
                 pos_margin=VIVE_POSITION_MARGIN_M,
                 rot_margin_deg=VIVE_ROTATION_MARGIN_DEG,
                 max_dt=VIVE_MAX_COMMAND_DT_S,
                 reject_linear=VIVE_REJECT_LINEAR_SPEED_M_S,
                 reject_angular_deg=VIVE_REJECT_ANGULAR_SPEED_DEG_S):
        self.last_sent_pose = np.asarray(initial_pose, dtype=float).copy()
        self.last_raw_pose = self.last_sent_pose.copy()
        self.last_sample_time = float(timestamp)
        self.rejected_count = 0
        self.max_linear = max_linear
        self.max_angular_deg = max_angular_deg
        self.pos_margin = pos_margin
        self.rot_margin_deg = rot_margin_deg
        self.max_dt = max_dt
        self.reject_linear = reject_linear
        self.reject_angular_deg = reject_angular_deg

    def reanchor(self, pose, timestamp):
        """Reset internal state to `pose` (call on teleop re-engage so the first
        post-pause frame starts clean instead of measuring a stale delta)."""
        self.last_sent_pose = np.asarray(pose, dtype=float).copy()
        self.last_raw_pose = self.last_sent_pose.copy()
        self.last_sample_time = float(timestamp)
        self.rejected_count = 0

    def clamp(self, candidate_pose, timestamp):
        """Slew-rate limit: move the last sent pose TOWARD `candidate` at a bounded
        linear/angular speed, and always return a pose (never hold/reject).

        This is what the REAL arm uses. It (a) evens out velocity -- the target
        advances at a smooth capped rate instead of the arm lagging then lurching
        to catch up; (b) keeps the streamed servo target from outrunning the arm,
        which is what left j4/j5 still rotating after 's' (post-stop coast); and
        (c) caps a glitch spike to one small step (corrected next frame) instead
        of a lurch. Unlike filter(), motion is continuous."""
        candidate_pose = np.asarray(candidate_pose, dtype=float)
        timestamp = float(timestamp)
        dt = min(max(timestamp - self.last_sample_time, 0.0), self.max_dt)
        self.last_sample_time = timestamp
        if not np.isfinite(candidate_pose).all():
            self.last_raw_pose = self.last_sent_pose.copy()
            return self.last_sent_pose.copy()   # drop NaN/inf, hold in place

        # Outlier reject: if the RAW tracker moved faster than a hand physically
        # can between this frame and the last, it's a VIVE glitch -> HOLD (don't
        # slew toward it). last_raw is still advanced, so once the tracker settles
        # (a real fast move) the next frame's per-frame speed is normal and passes.
        _dt = max(dt, 1e-3)
        raw_dist = float(np.linalg.norm(candidate_pose[:3, 3] - self.last_raw_pose[:3, 3]))
        raw_ang_deg = float(np.degrees(np.linalg.norm(
            _R.from_matrix(self.last_raw_pose[:3, :3].T @ candidate_pose[:3, :3]).as_rotvec())))
        if raw_dist / _dt > self.reject_linear or raw_ang_deg / _dt > self.reject_angular_deg:
            self.last_raw_pose = candidate_pose.copy()
            self.rejected_count += 1
            return self.last_sent_pose.copy()   # hold on glitch spike
        self.rejected_count = 0

        out = self.last_sent_pose.copy()
        # translation slew
        dpos = candidate_pose[:3, 3] - out[:3, 3]
        dist = float(np.linalg.norm(dpos))
        max_trans = self.max_linear * dt
        if dist > max_trans and dist > 1e-12:
            dpos *= max_trans / dist
        out[:3, 3] += dpos
        # rotation slew: rotvec of last->candidate, clipped to the angular cap
        relative_rotation = out[:3, :3].T @ candidate_pose[:3, :3]
        rotvec = _R.from_matrix(relative_rotation).as_rotvec()
        ang = float(np.linalg.norm(rotvec))
        max_rot = np.radians(self.max_angular_deg) * dt
        if ang > max_rot and ang > 1e-12:
            rotvec *= max_rot / ang
        out[:3, :3] = out[:3, :3] @ _R.from_rotvec(rotvec).as_matrix()

        self.last_sent_pose = out
        self.last_raw_pose = candidate_pose.copy()
        return out

    def filter(self, candidate_pose, timestamp):
        """Return (filtered_pose | None, translation_delta_m, rotation_delta_deg).
        None => candidate rejected as a glitch; caller should hold last pose."""
        candidate_pose = np.asarray(candidate_pose, dtype=float)
        timestamp = float(timestamp)
        dt = min(max(timestamp - self.last_sample_time, 0.0), self.max_dt)
        self.last_sample_time = timestamp
        translation_delta = np.linalg.norm(candidate_pose[:3, 3] - self.last_raw_pose[:3, 3])
        relative_rotation = self.last_raw_pose[:3, :3].T @ candidate_pose[:3, :3]
        cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
        rotation_delta_deg = np.degrees(np.arccos(cosine))
        translation_limit = self.max_linear * dt + self.pos_margin
        rotation_limit_deg = self.max_angular_deg * dt + self.rot_margin_deg
        accepted = (np.isfinite(candidate_pose).all()
                    and translation_delta <= translation_limit
                    and rotation_delta_deg <= rotation_limit_deg)
        if accepted:
            filtered_pose = self.last_sent_pose.copy()
            filtered_pose[:3, 3] += candidate_pose[:3, 3] - self.last_raw_pose[:3, 3]
            filtered_pose[:3, :3] = self.last_sent_pose[:3, :3] @ relative_rotation
            self.last_sent_pose = filtered_pose
            self.rejected_count = 0
        else:
            filtered_pose = None
            self.rejected_count += 1
        self.last_raw_pose = candidate_pose.copy()
        return filtered_pose, translation_delta, rotation_delta_deg
