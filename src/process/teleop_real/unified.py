"""teleop_real 세션 후처리: UNIFIED_MAPPING 포맷 저장 (+ viser viz).

/home/jisoo/data2/realtime_vlm/run/replay_recording.py 와 동일한 출력 계약.

입력  : {session}/arm/{time,position,action_qpos}.npy
        {session}/hand/{time,position,action}.npy   (postprocess_session 결과)
출력  : {session}/unified.npz       — UNIFIED_MAPPING 키 배열 (time 동기화)
        {session}/unified_meta.json — 단위/joint 순서/eef link 메타
        (선택) viser viewer

UNIFIED_MAPPING:
    state.eef_position       (T,3)  xarm6 FK (m)
    state.eef_rotation       (T,3)  axis-angle (rad)
    action.eef_position      (T,3)
    action.eef_rotation      (T,3)
    state.arm_joint_position (T,6)  xarm6 rad
    state.dexhand_position   (T,6)  inspire rad (qpos 순서)
    action.arm_joint_position(T,6)
    action.dexhand_position  (T,6)
    time                     (T,)   main PC pc_time

주의: unified.npz 의 dexhand 는 레퍼런스의 inspire_register_to_qpos
(normalized*limit, perm [5,4,3,2,1,0]) 를 그대로 사용. viser viz 는 paradex
parse_inspire 를 사용 (레퍼런스와 동일한 이중 컨벤션).
"""
import os
import sys
import json
import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R

from paradex.utils.path import shared_dir
from paradex.robot.utils import get_robot_urdf_path
from paradex.robot.inspire import parse_inspire

INSPIRE_QPOS_ORDER = ["thumb_yaw", "thumb_pitch", "index", "middle", "ring", "pinky"]
# URDF joint names, 1:1 with INSPIRE_QPOS_ORDER columns (and with
# visualize_unified.py's UNIFIED_HAND_URDF) -> all viewers stay consistent.
_QPOS_URDF = ["right_thumb_1_joint", "right_thumb_2_joint", "right_index_1_joint",
              "right_middle_1_joint", "right_ring_1_joint", "right_little_1_joint"]
INSPIRE_LIMITS_RAD = np.array([1.15, 0.55, 1.6, 1.6, 1.6, 1.6])
XARM_JOINT_ORDER = [f"joint{i+1}" for i in range(6)]


def inspire_register_to_qpos(reg_int):
    """raw inspire register (T,6) int 0~1000 -> qpos (T,6) rad in
    INSPIRE_QPOS_ORDER. Uses paradex parse_inspire (limit*(1-reg/1000)) so it
    matches the viser/overlay rendering (NOT the realtime_vlm reference, which
    omitted the inversion and made every finger look fully closed)."""
    assert reg_int.shape[-1] == 6
    return parse_inspire(np.asarray(reg_int, dtype=np.float64),
                         joint_order=_QPOS_URDF)


def time_sync_nearest(target_times, source_times, source_data):
    n, m = len(target_times), len(source_times)
    if m == 0:
        return np.zeros((n,) + source_data.shape[1:], dtype=source_data.dtype)
    out = np.zeros((n,) + source_data.shape[1:], dtype=source_data.dtype)
    j = 0
    for i in range(n):
        t = target_times[i]
        while j + 1 < m and abs(source_times[j + 1] - t) <= abs(source_times[j] - t):
            j += 1
        out[i] = source_data[j]
    return out


def cart_to_T(cart):
    """[x_mm,y_mm,z_mm, r,p,y(rad euler xyz)] -> (4,4) (m)."""
    Tm = np.eye(4)
    Tm[:3, 3] = np.asarray(cart[:3], dtype=np.float64) / 1000.0
    Tm[:3, :3] = R.from_euler("xyz", cart[3:]).as_matrix()
    return Tm


def reconstruct_action_qpos(action_cart, state_qpos, urdf_path, eef_link):
    """Rebuild action.arm_joint_position by IK on the commanded eef pose.
    Seeded from the measured state joint (teleop tracks closely => same IK
    branch + fast convergence). Falls back to random restarts if seed fails."""
    from paradex.robot.robot_wrapper import RobotWrapper
    robot = RobotWrapper(urdf_path)
    dof = robot.dof
    out = np.zeros((len(action_cart), dof), dtype=np.float64)
    prev = None
    n_ok = 0
    for i, cart in enumerate(action_cart):
        target = cart_to_T(cart)
        seed = state_qpos[i].astype(np.float64) if dof == state_qpos.shape[1] \
            else (prev if prev is not None else robot.q0)
        q, ok = robot.solve_ik(target, eef_link, q_init=seed,
                               max_iter=300, try_num=1)
        if not ok:  # seed failed -> allow random restarts
            q, ok = robot.solve_ik(target, eef_link, q_init=seed,
                                    max_iter=500, try_num=20)
        out[i] = q[:dof]
        prev = q.copy()
        n_ok += int(ok)
    print(f"[unified] action IK: {n_ok}/{len(action_cart)} converged")
    return out


def compute_eef(arm_qpos, urdf_path, eef_link):
    from paradex.robot.robot_wrapper import RobotWrapper
    robot = RobotWrapper(urdf_path)
    dof = robot.dof
    if dof != arm_qpos.shape[1]:
        print(f"[unified] URDF dof={dof} arm qpos cols={arm_qpos.shape[1]}; padding")
    poses = np.zeros((len(arm_qpos), 6), dtype=np.float64)
    for i, q in enumerate(arm_qpos):
        if dof == len(q):
            full_q = q.astype(np.float64)
        else:
            full_q = np.zeros(dof, dtype=np.float64)
            full_q[:len(q)] = q
        T = robot.compute_forward_kinematics(full_q, link_list=[eef_link])[eef_link]
        poses[i, :3] = T[:3, 3]
        poses[i, 3:] = R.from_matrix(T[:3, :3]).as_rotvec()
    return poses[:, :3], poses[:, 3:]


def resolve_session(arg):
    if os.path.isdir(arg):
        return os.path.abspath(arg)
    cand = os.path.join(shared_dir, arg)
    if os.path.isdir(cand):
        return cand
    sys.exit(f"[unified] session not found: {arg} (nor {cand})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session", help="absolute path OR path under shared_dir "
                                    "(e.g. teleop_test/task/.../2026-05-15_17-13-09)")
    p.add_argument("--arm", default="xarm")
    p.add_argument("--hand", default="inspire")
    p.add_argument("--eef_link", default="link6", help="EEF link in arm urdf")
    p.add_argument("--no_viser", action="store_true")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    sdir = resolve_session(args.session)
    if not os.path.isdir(os.path.join(sdir, "arm")) or \
       not os.path.isdir(os.path.join(sdir, "hand")):
        sys.exit(f"[unified] arm/ or hand/ missing in {sdir} "
                 f"(run postprocess_session first)")

    arm_urdf = get_robot_urdf_path(arm_name=args.arm)
    viz_urdf = get_robot_urdf_path(arm_name=args.arm, hand_name=args.hand)

    # ---- 1. load (postprocessed: arm/hand already on a common timeline) ----
    arm_time = np.load(os.path.join(sdir, "arm", "time.npy"))
    arm_position = np.load(os.path.join(sdir, "arm", "position.npy"))        # (T,6) rad (measured)
    arm_action_cart = np.load(os.path.join(sdir, "arm", "action.npy"))       # (T,6) [xyz mm, rpy rad] commanded eef
    hand_time = np.load(os.path.join(sdir, "hand", "time.npy"))
    hand_pos_reg = np.load(os.path.join(sdir, "hand", "position.npy"))       # (T,6) int
    hand_act_reg = np.load(os.path.join(sdir, "hand", "action.npy"))         # (T,6) int
    print(f"[unified] arm {arm_position.shape}  hand {hand_pos_reg.shape}")

    # ---- 2. inspire register -> qpos (rad) ----
    hand_pos_qpos = inspire_register_to_qpos(hand_pos_reg)
    hand_act_qpos = inspire_register_to_qpos(hand_act_reg)

    # ---- 3. hand -> arm timeline (no-op if already aligned) ----
    hand_pos_sync = time_sync_nearest(arm_time, hand_time, hand_pos_qpos)
    hand_act_sync = time_sync_nearest(arm_time, hand_time, hand_act_qpos)

    # ---- 4. eef ----
    # state: FK on measured joint (radians, correct).
    print("[unified] FK state.arm_joint_position ...")
    s_eef_p, s_eef_r = compute_eef(arm_position, arm_urdf, args.eef_link)
    # action: the teleop COMMAND is a Cartesian wrist pose, stored correctly in
    # arm/action.npy ([xyz mm, rpy rad]). Use it directly. (arm/action_qpos.npy
    # is NOT used for eef: pre-fix xArm get_inverse_kinematics mis-read radians
    # as deg AND returned deg -> doubly wrong, unrecoverable.)
    a_eef_p = arm_action_cart[:, :3].astype(np.float64) / 1000.0
    a_eef_r = R.from_euler("xyz", arm_action_cart[:, 3:]).as_rotvec()

    # action joint: rebuild via IK on the commanded eef (seeded from measured
    # state joint). Replaces the unrecoverable legacy arm/action_qpos.npy.
    print("[unified] IK action.arm_joint_position ...")
    arm_action_qpos = reconstruct_action_qpos(
        arm_action_cart, arm_position, arm_urdf, args.eef_link)

    # ---- 5. save unified.npz ----
    unified = {
        "state.eef_position": s_eef_p,
        "state.eef_rotation": s_eef_r,
        "action.eef_position": a_eef_p,
        "action.eef_rotation": a_eef_r,
        "state.arm_joint_position": arm_position.astype(np.float64),
        "state.dexhand_position": hand_pos_sync.astype(np.float64),
        "action.arm_joint_position": arm_action_qpos.astype(np.float64),  # IK of commanded eef
        "action.dexhand_position": hand_act_sync.astype(np.float64),
        "time": arm_time.astype(np.float64),
    }
    out_npz = os.path.join(sdir, "unified.npz")
    np.savez_compressed(out_npz, **unified)
    print(f"[unified] saved {out_npz}  ({len(arm_time)} frames)")

    meta = {
        "session_dir": sdir,
        "arm_urdf": arm_urdf,
        "arm_eef_link": args.eef_link,
        "arm_joint_order": XARM_JOINT_ORDER,
        "arm_joint_unit": "rad",
        "hand_qpos_order": INSPIRE_QPOS_ORDER,
        "hand_qpos_unit": "rad",
        "hand_qpos_limits_rad": INSPIRE_LIMITS_RAD.tolist(),
        "hand_register_order_in_raw_npy": [
            "pinky", "ring", "middle", "index", "thumb_pitch", "thumb_yaw"],
        "hand_register_unit_raw": "int 0-1000",
        "eef_position_unit": "m",
        "eef_rotation_format": "axis-angle (rad)",
        "action_eef_source": "arm/action.npy (commanded Cartesian wrist pose)",
        "action_arm_joint_position_source": (
            "offline IK (pinocchio solve_ik) on commanded eef, seeded from "
            "measured state joint. NOT the legacy arm/action_qpos.npy (which "
            "was unrecoverable due to the xArm get_inverse_kinematics unit bug)."),
        "time_unit": "main PC pc_time (s, time.time())",
        "n_frames": int(len(arm_time)),
    }
    out_meta = os.path.join(sdir, "unified_meta.json")
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[unified] saved {out_meta}")

    # ---- 6. viser (xarm_inspire.urdf: arm + hand) ----
    if args.no_viser:
        return
    try:
        from paradex.visualization.visualizer.viser import ViserViewer
        from paradex.visualization.robot import RobotModule
        from paradex.robot.inspire import parse_inspire
    except Exception as e:
        print(f"[unified] viser import failed ({e}) -- skip viz")
        return

    viz_module = RobotModule(viz_urdf)
    viz_joints = viz_module.get_joint_names()
    hand_joints_urdf = [j for j in viz_joints if j not in XARM_JOINT_ORDER]

    T = len(arm_time)
    full_qpos = np.zeros((T, len(viz_joints)), dtype=np.float64)
    for i, jn in enumerate(viz_joints):
        if jn in XARM_JOINT_ORDER:
            full_qpos[:, i] = arm_position[:, XARM_JOINT_ORDER.index(jn)]

    hand_reg_sync = time_sync_nearest(arm_time, hand_time, hand_pos_reg)
    hand_urdf_qpos = parse_inspire(hand_reg_sync.astype(np.float64),
                                   joint_order=hand_joints_urdf)
    for i, jn in enumerate(viz_joints):
        if jn in hand_joints_urdf:
            full_qpos[:, i] = hand_urdf_qpos[:, hand_joints_urdf.index(jn)]

    print(f"[unified] viser on :{args.port} (Ctrl+C to stop)")
    vis = ViserViewer(port_number=args.port)
    vis.add_robot("xarm_inspire", viz_urdf)
    vis.add_traj("session", {"xarm_inspire": full_qpos})
    vis.add_floor(height=0.0)
    vis.start_viewer()


if __name__ == "__main__":
    main()
