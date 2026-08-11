"""Replay a captured episode's ROBOT data (xArm + Inspire) in a Viser viewer.

Loads the recorded joint trajectory and plays it back on the xarm_inspire URDF
with a time slider. No pinocchio / IK needed (pure FK playback via yourdfpy).

Data layout (from CaptureSession):
    <session>/arm/position.npy   (T,6) arm joint angles [rad]      (state = actual)
    <session>/hand/position.npy  (T,6) Inspire actuator 0-1000     (state = actual)
    <session>/arm/action_qpos.npy + hand/action.npy                (--action = command)
  (top-level = postprocessed/aligned; falls back to <session>/raw/* if absent)

Usage:
    python src/dataset_acquisition/hri/visualize_capture.py \
        capture/hri_vive/zerodex_eval/2026-08-07_21-27-00 --port 8081
    #   --action  : show commanded trajectory instead of the actual (state) one
"""
import os
import argparse

import numpy as np

from paradex.utils.path import shared_dir
from paradex.visualization.visualizer.viser import ViserViewer

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
URDF = os.path.join(REPO, "rsc", "robot", "xarm_inspire.urdf")

# Inspire actuator (0-1000) -> URDF joint radians (inlined parse_inspire).
_HAND_JOINT_ORDER = ['right_thumb_1_joint', 'right_thumb_2_joint', 'right_index_1_joint',
                     'right_middle_1_joint', 'right_ring_1_joint', 'right_little_1_joint']
_INSPIRE_LIMIT = {'right_little_1_joint': 1.6, 'right_ring_1_joint': 1.6,
                  'right_middle_1_joint': 1.6, 'right_index_1_joint': 1.6,
                  'right_thumb_2_joint': 0.55, 'right_thumb_1_joint': 1.15}
# recorded 6-value order (RH56 ANGLE_ACT/SET): little,ring,middle,index,thumb_2,thumb_1
_INSPIRE_INPUT_ORDER = ['right_little_1_joint', 'right_ring_1_joint', 'right_middle_1_joint',
                        'right_index_1_joint', 'right_thumb_2_joint', 'right_thumb_1_joint']


def inspire_to_urdf(cmd):
    """(T,6) actuator 0-1000 -> (T,6) URDF joint radians in _HAND_JOINT_ORDER."""
    cmd = np.clip(np.asarray(cmd, dtype=float), 0.0, 1000.0)
    out = np.zeros((len(cmd), 6))
    for i, jn in enumerate(_INSPIRE_INPUT_ORDER):
        out[:, _HAND_JOINT_ORDER.index(jn)] = _INSPIRE_LIMIT[jn] * (1.0 - cmd[:, i] / 1000.0)
    return out


def couple_thumb_to_fingers(hand_v, couple_rot=False, gain=1.0):
    """Force the thumb to follow the 4 fingers (fixes the stuck-thumb data).

    cols: [little,ring,middle,index,thumb_2(bend),thumb_1(rot)], 0=closed 1000=open.
    Drive thumb_2 (bend) from the mean finger openness so the thumb grips when the
    fingers grip and opens when they open. gain>1 closes the thumb more than the
    fingers. Optionally couple thumb_1 (rotation/opposition) the same way."""
    hv = np.asarray(hand_v, dtype=float).copy()
    finger_open = hv[:, 0:4].mean(axis=1)                     # 0(closed)..1000(open)
    thumb = np.clip(1000.0 - gain * (1000.0 - finger_open), 0.0, 1000.0)
    hv[:, 4] = thumb                                          # thumb_2 (bend)
    if couple_rot:
        hv[:, 5] = thumb                                      # thumb_1 (rot/opposition)
    return hv


def _pick(session, rel):
    for base in (session, os.path.join(session, "raw")):
        p = os.path.join(base, rel)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"{rel} not found under {session} or {session}/raw")


def load_config_traj(session, action=False, couple_thumb=False, couple_rot=False, gain=1.0):
    arm_rel = "arm/action_qpos.npy" if action else "arm/position.npy"
    hand_rel = "hand/action.npy" if action else "hand/position.npy"
    arm_q = np.load(_pick(session, arm_rel)).astype(float)      # (Ta,6) rad
    hand_v = np.load(_pick(session, hand_rel)).astype(float)    # (Th,6) 0-1000
    T = min(len(arm_q), len(hand_v))
    arm_q, hand_v = arm_q[:T], hand_v[:T]
    if couple_thumb:
        hand_v = couple_thumb_to_fingers(hand_v, couple_rot=couple_rot, gain=gain)
    hand_q = inspire_to_urdf(hand_v)                            # (T,6) URDF rad
    return arm_q, hand_q, T


def build_full_qpos(viz_joints, arm_q, hand_q):
    T = len(arm_q)
    full = np.zeros((T, len(viz_joints)))
    for i in range(6):
        full[:, viz_joints.index(f"joint{i + 1}")] = arm_q[:, i]
    for k, jn in enumerate(_HAND_JOINT_ORDER):
        full[:, viz_joints.index(jn)] = hand_q[:, k]
    return full


def resolve(session):
    if os.path.isabs(session) and os.path.isdir(session):
        return session
    p = os.path.join(shared_dir, session)
    if os.path.isdir(p):
        return p
    if os.path.isdir(session):
        return os.path.abspath(session)
    raise FileNotFoundError(f"session not found: {session} (also tried {p})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="episode dir (abs, or relative under ~/shared_data)")
    ap.add_argument("--action", action="store_true",
                    help="replay the commanded trajectory instead of the actual state")
    ap.add_argument("--couple-thumb", action="store_true",
                    help="OVERRIDE the (stuck) recorded thumb: drive thumb bend from the mean "
                         "of the 4 fingers so it grips when they grip and opens when they open.")
    ap.add_argument("--couple-thumb1", action="store_true",
                    help="also couple thumb_1 (rotation/opposition) to the fingers.")
    ap.add_argument("--thumb-gain", type=float, default=1.0,
                    help="closure gain for the coupled thumb (>1 = thumb closes more than fingers).")
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()

    session = resolve(args.session)
    arm_q, hand_q, T = load_config_traj(session, action=args.action,
                                        couple_thumb=args.couple_thumb,
                                        couple_rot=args.couple_thumb1, gain=args.thumb_gain)
    tag = "ACTION (command)" if args.action else "STATE (actual)"
    if args.couple_thumb:
        tag += f" | THUMB COUPLED (gain={args.thumb_gain}{'+rot' if args.couple_thumb1 else ''})"
    print(f"[viz] {session}\n[viz] {T} frames | {tag}")

    from paradex.visualization.robot import RobotModule
    viz_joints = RobotModule(URDF).get_joint_names()
    full = build_full_qpos(viz_joints, arm_q, hand_q)

    vis = ViserViewer(port_number=args.port)
    vis.add_floor(height=0.0)
    vis.add_robot("robot", URDF)
    vis.add_traj("capture", {"robot": full})
    print(f"[viz] http://localhost:{args.port}  (use the time slider; Ctrl-C to quit)")
    vis.start_viewer()


if __name__ == "__main__":
    main()
