import argparse
import os
from typing import List, Optional

import numpy as np

from paradex.robot.utils import get_robot_urdf_path
from paradex.utils.path import rsc_path
from paradex.visualization.visualizer.viser import ViserViewer


def list_urdf_files() -> List[str]:
    """Return URDF files under rsc/robot (relative to that folder)."""
    base_dir = os.path.join(rsc_path, "robot")
    urdfs: List[str] = []
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if fname.endswith(".urdf"):
                rel_path = os.path.relpath(os.path.join(root, fname), base_dir)
                urdfs.append(rel_path)
    return sorted(urdfs)


def resolve_urdf_path(
    urdf_arg: Optional[str], arm: Optional[str], hand: Optional[str]
) -> str:
    """Resolve the URDF path from arguments."""
    if urdf_arg:
        # Allow passing either an absolute path or a path relative to rsc/robot.
        candidate = urdf_arg
        if not os.path.isabs(candidate):
            candidate = os.path.join(rsc_path, "robot", candidate)
        if not os.path.exists(candidate) and not candidate.endswith(".urdf"):
            # Convenience: add .urdf extension if omitted.
            extended = candidate + ".urdf"
            if os.path.exists(extended):
                candidate = extended
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"URDF not found: {candidate}")
        return candidate

    if arm or hand:
        if arm is None and hand is None:
            raise ValueError("Provide at least --arm or --hand, or use --urdf.")
        path = get_robot_urdf_path(arm_name=arm, hand_name=hand)
        if not os.path.exists(path):
            raise FileNotFoundError(f"URDF not found: {path}")
        return path

    raise ValueError("Provide --urdf or a combination of --arm/--hand.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize URDFs from rsc/robot with Viser."
    )
    parser.add_argument(
        "--urdf",
        type=str,
        help="URDF path (absolute or relative to rsc/robot).",
    )
    parser.add_argument("--arm", type=str, help="Arm name (used with --hand).")
    parser.add_argument(
        "--hand",
        type=str,
        help="Hand name. If provided without --arm, loads <hand>_float.urdf.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name to show in the viewer (defaults to URDF filename).",
    )
    parser.add_argument(
        "--list",
        dest="list_urdfs",
        action="store_true",
        help="List available URDFs under rsc/robot and exit.",
    )
    args = parser.parse_args()

    if args.list_urdfs:
        print("Available URDFs (relative to rsc/robot):")
        for path in list_urdf_files():
            print(f" - {path}")
        return

    urdf_path = resolve_urdf_path(args.urdf, args.arm, args.hand)
    robot_name = args.name or os.path.splitext(os.path.basename(urdf_path))[0]

    vis = ViserViewer()
    vis.add_robot(robot_name, urdf_path)

    # Keep viewer timeline happy with a single-frame trajectory at zero configuration.
    robot_module = vis.robot_dict[robot_name]
    zero_cfg = np.zeros(len(robot_module.urdf.joint_map))
    vis.add_traj("initial", {robot_name: zero_cfg[None, :]})

    print(f"Visualizing {urdf_path} as '{robot_name}'.")
    vis.start_viewer()


if __name__ == "__main__":
    main()
