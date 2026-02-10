import time
import numpy as np

from paradex.utils.system import network_info
from paradex.io.robot_controller.inspire_f1_controller import InspireF1Controller


def main():
    # Expect params under network_info['inspire_f1']['param']
    params = network_info["inspire_f1"]["param"]
    ic = InspireF1Controller(**params, tactile=True)

    ic.start("inspire_f1_motion")
    start_time = time.time()
    while time.time() - start_time < 10.0:
        phase = (time.time() - start_time) * 2 * np.pi * 0.2
        s = np.sin(phase)
        # Raw target ranges (RH56F1): [min, max]
        print(ic.get_data())
        ranges = np.array([
            [0, 1000],
            [0, 1000],
            [0, 1000],
            [0, 1000],
            [0, 1000],
            [0, 1000],
        ], dtype=np.float64)
        mid = ranges.mean(axis=1)
        amp = (ranges[:, 1] - ranges[:, 0]) / 2.0
        target = mid + amp * s
        ic.move(target.astype(np.float64))
        time.sleep(0.01)
    ic.end()


if __name__ == "__main__":
    main()
