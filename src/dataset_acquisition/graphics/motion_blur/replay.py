"""Replay pre-generated joint trajectory at a target speed.

Adapted from ~/mcc_minimal/collect_dynamic_data.py segment loop.
Drives `paradex.io.robot_controller.xarm_controller.XArmController`
via `controller.move(action_rad, is_servo=True)` — the controller's
internal 100Hz loop forwards each command with `set_servo_angle_j`.
"""
import time
import numpy as np

SAMPLE_HZ = 100
SAMPLE_DT = 1.0 / SAMPLE_HZ


def reset_to(controller, q_deg, speed_rad_s=0.35):
    q_rad = np.deg2rad(np.asarray(q_deg, dtype=float))
    controller.move(q_rad, is_servo=False, speed=speed_rad_s)


def replay_q_deg(controller, q_deg_traj, speed_deg_s):
    q_deg_traj = np.asarray(q_deg_traj, dtype=float)
    n_seg = len(q_deg_traj)

    for _ in range(20):
        controller.move(np.deg2rad(q_deg_traj[0]), is_servo=True)
        time.sleep(0.01)
    time.sleep(0.3)

    next_t = time.time()
    q_prev = q_deg_traj[0]
    for seg_i in range(1, n_seg):
        q_next = q_deg_traj[seg_i]
        delta = q_next - q_prev
        max_delta_deg = float(np.max(np.abs(delta)))
        move_time = max(SAMPLE_DT, max_delta_deg / speed_deg_s)
        n_steps = max(1, int(round(move_time * SAMPLE_HZ)))

        for step in range(n_steps):
            alpha = (step + 1) / n_steps
            q_cmd_rad = np.deg2rad(q_prev + alpha * delta)

            now = time.time()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
            next_t += SAMPLE_DT

            controller.move(q_cmd_rad, is_servo=True)

        q_prev = q_next
