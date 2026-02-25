import argparse
import os

import numpy as np

from paradex.io.robot_controller.xarm_controller import XArmController
from paradex.transforms.conversion import aa2mtx
from paradex.utils.system import network_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    arm = XArmController(**network_info["xarm"]["param"])
    idx = 0

    print("Type 'c' then Enter to save current qpos. Type 'q' then Enter to quit.")
    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == "q":
                break
            if cmd != "c":
                continue

            _, state = arm.arm.get_joint_states(is_radian=True)
            qpos = np.array(state[0])[:6]
            _, pos_aa = arm.arm.get_position_aa(is_radian=True)
            wrist_pos = aa2mtx(np.array(pos_aa))
            save_file = os.path.join(args.save_path, f"{idx + 50}_qpos.npy")
            aa_save_file = os.path.join(args.save_path, f"{idx + 50}_aa.npy")
            np.save(save_file, qpos)
            np.save(aa_save_file, wrist_pos)
            print(f"Saved qpos[{idx}] -> {save_file}")
            print(f"Saved aa[{idx}] -> {aa_save_file}")
            idx += 1
    finally:
        arm.end()
        print("Recorder closed.")


if __name__ == "__main__":
    main()
