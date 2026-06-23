#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import numpy as np

from paradex.visualization.visualizer.viser import ViserViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple URDF visualizer with viser")
    parser.add_argument("urdf_path", type=str, help="Path to URDF file")
    parser.add_argument("--robot-name", type=str, default="robot", help="Robot node name")
    parser.add_argument(
        "--no-arm-meshes",
        action="store_true",
        help="Hide non-hand arm meshes (uses existing ViserRobotModule option)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urdf_path = Path(args.urdf_path).expanduser().resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    viewer = ViserViewer(scene_title=f"URDF: {urdf_path.name}")
    viewer.add_robot(
        name=args.robot_name,
        urdf_path=str(urdf_path),
        include_arm_meshes=not args.no_arm_meshes,
    )

    robot = viewer.robot_dict[args.robot_name]
    joint_names = list(robot.urdf.get_joint_names())
    joint_limits = robot.urdf.get_joint_limits()

    if joint_names:
        sliders = {}
        for joint_name in joint_names:
            lower, upper = joint_limits.get(joint_name, (-np.pi, np.pi))
            if lower is None or upper is None or not np.isfinite(lower) or not np.isfinite(upper):
                lower, upper = -np.pi, np.pi
            if upper <= lower:
                lower, upper = -np.pi, np.pi

            sliders[joint_name] = viewer.server.gui.add_slider(
                label=joint_name,
                min=float(lower),
                max=float(upper),
                step=0.001,
                initial_value=0.0,
            )

        def _on_update(_: object) -> None:
            cfg = np.array([sliders[name].value for name in joint_names], dtype=float)
            robot.update_cfg(cfg)

        for slider in sliders.values():
            slider.on_update(_on_update)

    print(f"Serving viser viewer for: {urdf_path}")
    print("Open the URL printed by viser (typically http://localhost:8080)")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
