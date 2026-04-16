import argparse
import os

import numpy as np

from paradex.utils.path import rsc_path
from paradex.visualization.robot import RobotModule
from paradex.visualization.visualizer.viser import ViserViewer


def to_2d(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape={arr.shape}")
    return arr


def match_urdf_dof(qpos: np.ndarray, dof: int, name: str) -> np.ndarray:
    if qpos.shape[1] < dof:
        raise ValueError(
            f"{name} has {qpos.shape[1]} joints, but URDF requires at least {dof}."
        )
    if qpos.shape[1] > dof:
        print(
            f"[WARN] {name} has {qpos.shape[1]} joints, truncating to first {dof} joints to match URDF."
        )
    return qpos[:, :dof]


def build_pose(x_offset: float) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[0, 3] = float(x_offset)
    return pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize action_qpos.npy and position.npy together with xarm URDF "
            "to check alignment in 3D."
        )
    )
    parser.add_argument(
        "--base_path",
        required=True,
        type=str,
        help="Base path that contains raw/arm/action_qpos.npy and raw/arm/position.npy",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=os.path.join(rsc_path, "robot", "xarm", "xarm.urdf"),
        help="URDF path (default: rsc/robot/xarm/xarm.urdf)",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=-1, help="End frame index (exclusive, -1 means all)")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--fps", type=float, default=20.0, help="Viewer playback FPS")
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="X offset between action/state robots. 0 means full overlay.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arm_dir = os.path.join(args.base_path, "raw", "arm")
    action_path = os.path.join(arm_dir, "action_qpos.npy")
    state_path = os.path.join(arm_dir, "position.npy")

    if not os.path.exists(action_path):
        raise FileNotFoundError(f"Missing file: {action_path}")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing file: {state_path}")
    if not os.path.exists(args.urdf):
        raise FileNotFoundError(f"URDF not found: {args.urdf}")

    action_qpos = to_2d(np.load(action_path), "action_qpos")
    state_qpos = to_2d(np.load(state_path), "position")

    num_frames = min(len(action_qpos), len(state_qpos))
    if len(action_qpos) != len(state_qpos):
        print(
            f"[WARN] Frame count mismatch: action={len(action_qpos)}, state={len(state_qpos)}. "
            f"Using first {num_frames} frames."
        )

    action_qpos = action_qpos[:num_frames]
    state_qpos = state_qpos[:num_frames]

    robot = RobotModule(args.urdf)
    dof = robot.get_num_joints()
    action_qpos = match_urdf_dof(action_qpos, dof, "action_qpos")
    state_qpos = match_urdf_dof(state_qpos, dof, "position")

    end = num_frames if args.end < 0 else min(args.end, num_frames)
    start = max(0, args.start)
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if start >= end:
        raise ValueError(f"Invalid frame range: start={start}, end={end}")

    action_qpos = action_qpos[start:end:args.stride]
    state_qpos = state_qpos[start:end:args.stride]

    diff = action_qpos - state_qpos
    per_frame_l2 = np.linalg.norm(diff, axis=1)
    print(f"[INFO] base_path: {args.base_path}")
    print(f"[INFO] urdf: {args.urdf}")
    print(f"[INFO] dof used: {dof}")
    print(f"[INFO] frames visualized: {len(action_qpos)}")
    print(f"[INFO] per-joint MAE: {np.mean(np.abs(diff), axis=0)}")
    print(f"[INFO] global RMSE: {np.sqrt(np.mean(diff * diff)):.6f}")
    print(
        "[INFO] frame L2 error stats: "
        f"min={per_frame_l2.min():.6f}, mean={per_frame_l2.mean():.6f}, max={per_frame_l2.max():.6f}"
    )

    vis = ViserViewer(scene_title="xarm_action_state_align")

    half_offset = args.offset * 0.5
    vis.add_robot("action", args.urdf, pose=build_pose(-half_offset))
    vis.add_robot("state", args.urdf, pose=build_pose(+half_offset))

    vis.change_color("action", (1.0, 0.2, 0.2, 0.60))
    vis.change_color("state", (0.2, 1.0, 0.2, 0.60))

    vis.add_traj(
        "align_compare",
        {
            "action": action_qpos,
            "state": state_qpos,
        },
    )
    vis.gui_framerate.value = float(args.fps)

    print("[INFO] action=red, state=green (both semi-transparent)")
    print("[INFO] Launching viewer... (Ctrl+C to stop)")
    vis.start_viewer()


if __name__ == "__main__":
    main()
