import os
import argparse
import numpy as np

from paradex.utils.path import shared_dir
from paradex.robot.utils import get_robot_urdf_path
from paradex.robot.inspire import parse_inspire
from paradex.visualization.visualizer.viser import ViserViewer


parser = argparse.ArgumentParser()
parser.add_argument("--demo_path", type=str, required=True,
                    help="path under shared_dir, e.g. teleop_test/01/2026-05-15_14-23-00")
parser.add_argument("--arm", type=str, default="xarm")
parser.add_argument("--hand", type=str, default="inspire")
args = parser.parse_args()

demo_dir = os.path.join(shared_dir, args.demo_path)
arm_path = os.path.join(demo_dir, "arm", "position.npy")
hand_path = os.path.join(demo_dir, "hand", "position.npy")

if not os.path.exists(arm_path) or not os.path.exists(hand_path):
    raise FileNotFoundError(
        f"Missing synced data. Expected {arm_path} and {hand_path}. "
        f"Run postprocess_session() first or pass the post-processed path."
    )

arm_pos = np.load(arm_path)
hand_pos = np.load(hand_path)

if args.hand == "inspire":
    hand_pos = parse_inspire(hand_pos)
elif args.hand == "allegro":
    pass
else:
    raise ValueError(f"Unsupported hand: {args.hand}")

assert len(arm_pos) == len(hand_pos), \
    f"arm and hand frame counts differ: {len(arm_pos)} vs {len(hand_pos)}. " \
    f"Did you run postprocess?"

traj = np.concatenate([arm_pos, hand_pos], axis=1)
print(f"Loaded {len(traj)} frames, dof = {traj.shape[1]}")

vis = ViserViewer()
vis.add_robot("robot", get_robot_urdf_path(args.arm, args.hand))
vis.add_traj("teleop", {"robot": traj})
vis.add_floor(-0.0525)
print("Viser running at http://0.0.0.0:8080")
vis.start_viewer()
