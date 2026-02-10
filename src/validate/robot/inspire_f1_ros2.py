import time
import numpy as np

from paradex.io.robot_controller.inspire_f1_controller_ros2 import InspireF1ControllerROS2


def main():
    ctrl = InspireF1ControllerROS2(hand_side="left")

    start_time = time.time()
    try:
        while time.time() - start_time < 10.0:
            phase = (time.time() - start_time) * 2 * np.pi * 0.2
            s = np.sin(phase)
            ranges = np.array([
                [900, 1740],
                [900, 1740],
                [900, 1740],
                [900, 1740],
                [1100, 1350],
                [600, 1800],
            ], dtype=np.float64)
            mid = ranges.mean(axis=1)
            amp = (ranges[:, 1] - ranges[:, 0]) / 2.0
            target = mid + amp * s
            ctrl.move(target.astype(np.float64))
            print(ctrl.get_data())
            time.sleep(0.01)
    finally:
        ctrl.end()


if __name__ == "__main__":
    main()
