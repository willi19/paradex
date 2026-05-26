"""XArm base joint(joint1) 도리도리 validate.

현재 자세에서 joint1만 ±60°로 부드럽게 좌우 흔든다. 나머지 joint는 home 값을
매 iteration 동일하게 보내 drift를 막는다.

Usage:
    python src/validate/robot/xarm_base_wiggle.py
"""

import time

import numpy as np

from paradex.io.robot_controller import get_arm


AMPLITUDE_DEG = 60.0
PERIOD_S = 6.0
DURATION_S = 120.0
LOOP_HZ = 60.0


def main():
    arm = get_arm("xarm")
    arm.connect_event.wait()

    home_qpos = np.array(arm.get_data()["qpos"], dtype=np.float64).copy()
    amp = np.deg2rad(AMPLITUDE_DEG)
    dt = 1.0 / LOOP_HZ

    print(f"[wiggle] home qpos (rad): {np.round(home_qpos, 4).tolist()}")
    print(f"[wiggle] joint1 ±{AMPLITUDE_DEG}°, period {PERIOD_S}s, duration {DURATION_S}s")

    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            if t > DURATION_S:
                break
            target = home_qpos.copy()
            target[0] = home_qpos[0] + amp * np.sin(2.0 * np.pi * t / PERIOD_S)
            arm.move(target, is_servo=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("[wiggle] interrupted")
    finally:
        print("[wiggle] returning to home pose ...")
        arm.move(home_qpos, is_servo=False)
        arm.end(False)
        print("[wiggle] done")


if __name__ == "__main__":
    main()
