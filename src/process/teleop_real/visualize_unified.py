"""이미 저장된 unified.npz 를 viser 로 재생.

unified.py 가 만든 {session}/unified.npz 의 arm/dexhand joint 배열을
xarm_inspire URDF 에 그대로 매핑해서 재생. state(실제) 와 action(명령) 을
함께 띄울 수 있음.

unified.npz 의 dexhand_position 은 INSPIRE_QPOS_ORDER 컬럼:
    [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
parse_inspire 의 joint_order 와 1:1 대응:
    [right_thumb_1_joint, right_thumb_2_joint, right_index_1_joint,
     right_middle_1_joint, right_ring_1_joint, right_little_1_joint]
mimic joint (thumb_3/4, *_2) 는 yourdfpy 가 자동 처리 -> 0 으로 둠.
"""
import os
import sys
import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R

from paradex.utils.path import shared_dir
from paradex.robot.utils import get_robot_urdf_path
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.visualization.robot import RobotModule

XARM_JOINT_ORDER = [f"joint{i+1}" for i in range(6)]
UNIFIED_HAND_URDF = [
    "right_thumb_1_joint", "right_thumb_2_joint", "right_index_1_joint",
    "right_middle_1_joint", "right_ring_1_joint", "right_little_1_joint",
]


def resolve_session(arg):
    if os.path.isdir(arg) or os.path.isfile(arg):
        return os.path.abspath(arg)
    cand = os.path.join(shared_dir, arg)
    if os.path.exists(cand):
        return cand
    sys.exit(f"[viz_unified] not found: {arg} (nor {cand})")


def build_full_qpos(viz_joints, arm_jpos, hand_qpos):
    """(T,6) arm + (T,6) dexhand(unified order) -> (T, n_viz_joints)."""
    T = len(arm_jpos)
    full = np.zeros((T, len(viz_joints)), dtype=np.float64)
    for i, jn in enumerate(viz_joints):
        if jn in XARM_JOINT_ORDER:
            full[:, i] = arm_jpos[:, XARM_JOINT_ORDER.index(jn)]
        elif jn in UNIFIED_HAND_URDF:
            full[:, i] = hand_qpos[:, UNIFIED_HAND_URDF.index(jn)]
    return full


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session", help="session dir OR unified.npz path "
                                   "(abs or under shared_dir)")
    p.add_argument("--arm", default="xarm")
    p.add_argument("--hand", default="inspire")
    p.add_argument("--show_action", action="store_true",
                   help="also overlay the action(commanded) robot")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    path = resolve_session(args.session)
    npz_path = path if path.endswith(".npz") else os.path.join(path, "unified.npz")
    if not os.path.isfile(npz_path):
        sys.exit(f"[viz_unified] unified.npz not found: {npz_path}\n"
                 f"  run: python src/process/teleop_real/unified.py <session> --no_viser")

    d = np.load(npz_path)
    T = len(d["time"])
    print(f"[viz_unified] {npz_path}  {T} frames  "
          f"span {d['time'][-1]-d['time'][0]:.1f}s")

    viz_urdf = get_robot_urdf_path(arm_name=args.arm, hand_name=args.hand)
    viz_joints = RobotModule(viz_urdf).get_joint_names()

    state_q = build_full_qpos(viz_joints,
                              d["state.arm_joint_position"],
                              d["state.dexhand_position"])

    vis = ViserViewer(port_number=args.port)
    vis.add_robot("state", viz_urdf)
    traj = {"state": state_q}
    obj_traj = {}

    if args.show_action:
        # action.arm_joint_position is now valid (offline IK of commanded eef,
        # seeded from state in unified.py) -> render a real action robot.
        action_q = build_full_qpos(viz_joints,
                                   d["action.arm_joint_position"],
                                   d["action.dexhand_position"])
        vis.add_robot("action", viz_urdf)
        traj["action"] = action_q
        print("[viz_unified] overlaying state + action robot")

    vis.add_traj("unified", traj, obj_traj)
    vis.add_floor(height=0.0)
    print(f"[viz_unified] http://localhost:{args.port}  (Ctrl+C to stop)")
    vis.start_viewer()


if __name__ == "__main__":
    main()
