"""Dry-run PREVIEW of the robot under VIVE+MANUS teleop, WITHOUT commanding the
real robot.

Default (Cartesian) mode mirrors how the real xArm is actually driven: it is
Cartesian servo'd (the robot does IK internally). So the preview shows the
retargeted WRIST target pose directly (a frame gizmo) with the Inspire hand
rendered floating at that pose — no per-frame IK. Pass --show-arm to also solve
IK and render the full xArm mesh (heavier, can stall near singularities).

The hand is driven from the MANUS glove's ergonomics angles
(inspire_from_manus_ergonomics), which stay in [0,1000] and move the thumb
reliably, instead of the geometric inspire() formula whose thumb output saturates.

Prerequisites (same as the input viewer; no robot):
  - source /opt/ros/humble/setup.bash
  - source <vive-teleop>/ros2_ws/install/setup.bash
  - VIVE + BOTH MANUS glove publishers running

Run:  python src/dataset_acquisition/hri/preview_robot_teleop.py --port 8081
      (add --show-arm for the full IK arm)
"""
import sys
import time
import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R

# ROS 2 ships a NumPy-1 pinocchio that crashes under NumPy>=2 and shadows the
# conda build (because /opt/ros is earlier on sys.path). Import the conda
# pinocchio first so it is cached, then restore ROS paths so rclpy still
# resolves for the receiver. Must run before any `paradex.robot` import.
_ros = [p for p in sys.path if "/opt/ros/" in p]
for _p in _ros:
    sys.path.remove(_p)
import pinocchio  # noqa: F401  conda build, NumPy-2 compatible
for _p in _ros:
    sys.path.append(_p)

from paradex.io.teleop.vive.receiver import ViveManusROSReceiver
from paradex.retargetor.unimanual import Retargetor
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.robot.utils import get_robot_urdf_path
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.visualization.robot import RobotModule

ARM = "xarm"
HAND = "inspire"
EE_LINK = "link6"

# Wrist command safety limiter is shared with the real teleop path (Retargetor)
# so preview and capture behave identically. See paradex/retargetor/limiter.py.
from paradex.retargetor.limiter import PoseCommandLimiter as _PoseCommandLimiter

# --- Inspire actuator (0-1000) -> URDF joint radians (inlined parse_inspire) ---
_HAND_JOINT_ORDER = ['right_thumb_1_joint', 'right_thumb_2_joint', 'right_index_1_joint',
                     'right_middle_1_joint', 'right_ring_1_joint', 'right_little_1_joint']
_INSPIRE_LIMIT = {'right_little_1_joint': 1.6, 'right_ring_1_joint': 1.6,
                  'right_middle_1_joint': 1.6, 'right_index_1_joint': 1.6,
                  'right_thumb_2_joint': 0.55, 'right_thumb_1_joint': 1.15}
# command index i -> joint name (RH56 ANGLE_SET order)
_INSPIRE_INPUT_ORDER = ['right_little_1_joint', 'right_ring_1_joint', 'right_middle_1_joint',
                        'right_index_1_joint', 'right_thumb_2_joint', 'right_thumb_1_joint']

# --- MANUS ergonomics -> Inspire command (ported from paraoffice-vive) ---
_INSPIRE_FINGER_ANGLE_RANGE_DEG = 176.7 - 19.0
_INSPIRE_THUMB_BEND_RANGE_DEG = 53.6 - (-13.0)
_INSPIRE_THUMB_ROTATE_RANGE_DEG = 165.0 - 90.0
_INSPIRE_THUMB_EXTRA_BEND_GAIN = 3.0
_INSPIRE_THUMB_EXTRA_BEND_COMMAND = -500.0
_INSPIRE_THUMB_SECONDARY_GAIN = 0.8


def _open_to_closed_command(flexion_deg, range_deg):
    normalized = np.clip(float(flexion_deg) / float(range_deg), 0.0, 1.0)
    return (1.0 - normalized) * 1000.0


def inspire_from_manus_ergonomics(ergonomics):
    """Map MANUS ergonomic angles to Inspire ANGLE_SET order (6,), each in [0,1000]."""
    required = []
    for finger in ("Pinky", "Ring", "Middle", "Index"):
        required += [f"{finger}MCPStretch", f"{finger}PIPStretch", f"{finger}DIPStretch"]
    required += ["ThumbMCPStretch", "ThumbMCPSpread"]
    missing = [n for n in required if n not in ergonomics]
    if missing:
        raise ValueError("MANUS ergonomics missing: " + ", ".join(missing))

    command = np.zeros(6, dtype=np.float64)
    for index, finger in enumerate(("Pinky", "Ring", "Middle", "Index")):
        flexion = sum(float(ergonomics[f"{finger}{j}Stretch"]) for j in ("MCP", "PIP", "DIP"))
        command[index] = _open_to_closed_command(flexion, _INSPIRE_FINGER_ANGLE_RANGE_DEG)
    command[4] = np.clip(
        (np.clip(float(ergonomics["ThumbMCPSpread"]) / _INSPIRE_THUMB_ROTATE_RANGE_DEG, 0.0, 1.0)
         * 1000.0 * _INSPIRE_THUMB_EXTRA_BEND_GAIN) + _INSPIRE_THUMB_EXTRA_BEND_COMMAND,
        0.0, 1000.0)
    command[5] = np.clip(
        (1000.0 - _open_to_closed_command(ergonomics["ThumbMCPStretch"], _INSPIRE_THUMB_BEND_RANGE_DEG))
        * _INSPIRE_THUMB_SECONDARY_GAIN,
        0.0, 1000.0)
    return command


def inspire_cmd_to_joints(cmd):
    """(6,) inspire command 0-1000 -> dict-ordered 6 URDF joint radians (_HAND_JOINT_ORDER)."""
    cmd = np.clip(np.asarray(cmd, dtype=float), 0.0, 1000.0)
    q = np.zeros(6)
    for i, jn in enumerate(_INSPIRE_INPUT_ORDER):
        q[_HAND_JOINT_ORDER.index(jn)] = _INSPIRE_LIMIT[jn] * (1.0 - cmd[i] / 1000.0)
    return q


def _matrix_to_wxyz(matrix):
    q = R.from_matrix(matrix[:3, :3]).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


def _hand_joints(data, retgt_hand_action):
    """Prefer MANUS ergonomics (reliable thumb); fall back to geometric inspire()."""
    ergo = (data.get("ergonomics") or {}).get("Right") if data else None
    if ergo:
        try:
            return inspire_cmd_to_joints(inspire_from_manus_ergonomics(ergo)), True
        except (ValueError, KeyError):
            pass
    return inspire_cmd_to_joints(retgt_hand_action), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand-side", default="right", choices=["right"])
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--scale", "--hand-scale", dest="scale", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--home-qpos", type=str,
                    default="-0.06806784,0.02617994,-1.40499005,"
                            "-0.04014257,0.08552113,-1.32819556",
                    help="6 arm joint radians anchoring the preview at the REAL xArm6 home.")
    ap.add_argument("--hand-smooth", type=float, default=0.6,
                    help="EMA factor for hand joints (0=raw, ->1=smoother).")
    ap.add_argument("--no-limit", dest="limit", action="store_false",
                    help="Disable the wrist velocity/noise rejection limiter.")
    ap.set_defaults(limit=True)
    ap.add_argument("--show-arm", action="store_true",
                    help="Also solve IK and render the full xArm mesh (heavier).")
    args = ap.parse_args()
    beta = float(np.clip(args.hand_smooth, 0.0, 0.99))

    # Anchor at the real robot home (pinocchio FK gives the wrist start pose).
    q_home = np.array([float(x) for x in args.home_qpos.split(",")], dtype=float)
    ik = RobotWrapper(get_robot_urdf_path(arm_name=ARM))
    assert len(q_home) == ik.model.nq, f"--home-qpos needs {ik.model.nq} values"
    init_ee = ik.compute_forward_kinematics(q_home, [EE_LINK])[EE_LINK]

    # Preview runs its OWN external limiter below (for the delta/rejection debug
    # log), so disable the Retargetor's internal one to avoid double-filtering.
    retgt = Retargetor(arm_name=ARM, hand_name=HAND, translation_scale=args.scale,
                       limit_wrist=False)
    retgt.start(init_ee)

    receiver = ViveManusROSReceiver(hand_side=args.hand_side)
    vis = ViserViewer(port_number=args.port)
    vis.add_floor(height=0.0)

    if args.show_arm:
        viz_urdf = get_robot_urdf_path(arm_name=ARM, hand_name=HAND)   # xarm_inspire.urdf
    else:
        viz_urdf = get_robot_urdf_path(hand_name=HAND)                 # inspire_float.urdf
    viz_module = RobotModule(viz_urdf)
    viz_joints = viz_module.get_joint_names()
    _lim = viz_module.get_joint_limits()
    lo = np.array([_lim.get(n, (-np.inf, np.inf))[0] for n in viz_joints])
    hi = np.array([_lim.get(n, (-np.inf, np.inf))[1] for n in viz_joints])
    hand_idx = [viz_joints.index(n) for n in _HAND_JOINT_ORDER]

    vis.add_robot("robot", viz_urdf, pose=init_ee if not args.show_arm else None)
    robot = vis.robot_dict["robot"]
    if args.show_arm:
        arm_idx = [viz_joints.index(f"joint{i}") for i in range(1, 7)]
    status = vis.server.gui.add_text("status", initial_value="waiting for data...")
    print(f"Robot teleop preview ({'ARM+IK' if args.show_arm else 'Cartesian wrist+hand'}): "
          f"http://localhost:{args.port}  (Ctrl-C to quit)")

    def render(target_pose, q_hand):
        cfg = np.zeros(len(viz_joints))
        for k, idx in enumerate(hand_idx):
            cfg[idx] = q_hand[k]
        if args.show_arm:
            nonlocal q_arm
            q_sol, ok = ik.solve_ik(target_pose, EE_LINK, q_init=q_arm,
                                    max_iter=1000, tol=1e-4, try_num=1)
            q_arm = q_sol
            for k, idx in enumerate(arm_idx):
                cfg[idx] = q_arm[k]
            robot.update_cfg(np.clip(cfg, lo, hi))
            return ok
        # Cartesian: place the floating hand's root at the wrist target pose.
        robot.update_cfg(np.clip(cfg, lo, hi))
        if hasattr(robot, "_visual_root_frame"):
            robot._visual_root_frame.position = target_pose[:3, 3]
            robot._visual_root_frame.wxyz = _matrix_to_wxyz(target_pose)
        vis.server.scene.add_frame("target_wrist", axes_length=0.08, axes_radius=0.005,
                                   position=target_pose[:3, 3], wxyz=_matrix_to_wxyz(target_pose))
        return None

    q_arm = q_home.copy()
    q_hand_ema = None
    limiter = _PoseCommandLimiter(init_ee, time.monotonic()) if args.limit else None
    # idle render at home
    render(init_ee, inspire_cmd_to_joints(np.full(6, 1000.0)))

    period = 1.0 / max(args.fps, 1.0)
    last_log = 0.0
    n_none = 0
    try:
        while True:
            now = time.time()
            data = receiver.get_data()
            right = data.get("Right") if data else None
            if right is None:
                n_none += 1
                status.value = "no fresh data — check VIVE + BOTH MANUS gloves and ROS sourcing"
                if now - last_log > 1.0:
                    print(f"[preview] NO fresh data (None x{n_none}).")
                    last_log = now
                time.sleep(period)
                continue

            arm_action, retgt_hand = retgt.get_action(data)

            if limiter is not None:
                target_pose, tdelta, rdelta = limiter.filter(arm_action, time.monotonic())
            else:
                target_pose, tdelta, rdelta = arm_action, 0.0, 0.0
            rejected = target_pose is None
            if rejected:
                target_pose = limiter.last_sent_pose if limiter is not None else arm_action

            q_hand_raw, used_ergo = _hand_joints(data, retgt_hand)
            q_hand_ema = q_hand_raw.copy() if q_hand_ema is None else \
                beta * q_hand_ema + (1.0 - beta) * q_hand_raw

            ok = render(target_pose, q_hand_ema)

            status.value = (f"streaming | hand={'ergonomics' if used_ergo else 'geometric'}"
                            f"{' | IK ok:'+str(bool(ok)) if args.show_arm else ''}")
            if now - last_log > 1.0:
                print(f"[preview] fresh | cmd delta {tdelta*1000:.1f}mm/{rdelta:.1f}deg"
                      f"{' [REJECTED]' if rejected else ''} | hand src="
                      f"{'ergonomics' if used_ergo else 'geometric'}"
                      f"{' | IK ok '+str(bool(ok)) if args.show_arm else ''}")
                last_log = now
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.end()


if __name__ == "__main__":
    main()
