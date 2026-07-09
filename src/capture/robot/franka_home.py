"""Move the Franka (FR3) to its default home / "ready" joint configuration.

Uses the fr3_arm_controller (JointTrajectoryController) via the
follow_joint_trajectory action -- so the franka_ros2 stack must be up with
fr3_arm_controller ACTIVE (if gravity_compensation is active from a teaching
session, switch back to fr3_arm_controller first).

    ros2 control switch_controllers -c /controller_manager \
        --activate fr3_arm_controller --deactivate gravity_compensation_example_controller

The move is joint-space (limit-safe: interpolates between two in-range poses) and
slow by default to avoid velocity-violation reflexes.

    python src/capture/robot/franka_home.py [--move_time 8]
"""

import argparse
import numpy as np

from paradex.io.robot_controller.franka_controller import (
    FrankaController, FRANKA_HOME_QPOS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_size", type=float, default=0.15,
                        help="rad per waypoint (smaller = slower/finer)")
    parser.add_argument("--step_time", type=float, default=0.5,
                        help="seconds per waypoint (larger = slower)")
    args = parser.parse_args()

    controller = FrankaController(step_size=args.step_size, step_time=args.step_time)
    print(f"Moving to home {np.round(FRANKA_HOME_QPOS, 4).tolist()} ...")
    controller.move(np.array(FRANKA_HOME_QPOS, dtype=float), is_servo=False)
    print("Reached home.")
    controller.end()


if __name__ == "__main__":
    main()
