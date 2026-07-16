#!/usr/bin/env python3
"""Record and inspect Wuji joint commands and measured joint states.

Run this in a separate ROS2-sourced terminal before starting Paradex teleop.
Press Ctrl+C to save all samples and write a jump/tracking report.
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


DOF = 20
FINGERS = ("thumb", "index", "middle", "ring", "little")
ROLES = ("mcp_abd", "mcp_flex", "pip", "dip")
JOINT_LABELS = tuple(
    "{}_{}".format(finger, role) for finger in FINGERS for role in ROLES
)
DRIVER_JOINT_NAMES = tuple(
    "finger{}_joint{}".format(finger, joint)
    for finger in range(1, 6)
    for joint in range(1, 5)
)


def sensor_qos():
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=100,
    )


def ros_stamp(msg):
    stamp = msg.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def normalize_name(name):
    for prefix in ("right_", "left_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def ordered_position(msg):
    """Return driver finger-major order even when JointState names are reordered."""

    if len(msg.position) != DOF:
        return None
    if len(msg.name) == len(msg.position):
        by_name = {
            normalize_name(name): float(value)
            for name, value in zip(msg.name, msg.position)
        }
        if all(name in by_name for name in DRIVER_JOINT_NAMES):
            return np.asarray(
                [by_name[name] for name in DRIVER_JOINT_NAMES], dtype=np.float64
            )
    return np.asarray(msg.position, dtype=np.float64)


class WujiSignalRecorder(Node):
    def __init__(self, command_topic, state_topic, print_hz):
        super().__init__("paradex_wuji_signal_check")
        self.command_topic = command_topic
        self.state_topic = state_topic
        self.print_period = 0.0 if print_hz <= 0.0 else 1.0 / print_hz
        self.last_print = 0.0
        self.command = []
        self.command_time = []
        self.command_ros_time = []
        self.state = []
        self.state_time = []
        self.state_ros_time = []
        self.state_names = []
        self.create_subscription(
            JointState, command_topic, self.on_command, sensor_qos()
        )
        self.create_subscription(JointState, state_topic, self.on_state, sensor_qos())

    def on_command(self, msg):
        values = ordered_position(msg)
        if values is None:
            self.get_logger().warning(
                "Ignoring command with {} positions".format(len(msg.position))
            )
            return
        self.command.append(values)
        self.command_time.append(time.time())
        self.command_ros_time.append(ros_stamp(msg))
        now = time.monotonic()
        if self.print_period and now - self.last_print >= self.print_period:
            self.last_print = now
            text = " ".join(
                "{}={:+.3f}".format(name, value)
                for name, value in zip(JOINT_LABELS, values)
            )
            print("[command] {}".format(text), flush=True)

    def on_state(self, msg):
        values = ordered_position(msg)
        if values is None:
            self.get_logger().warning(
                "Ignoring state with {} positions".format(len(msg.position))
            )
            return
        self.state.append(values)
        self.state_time.append(time.time())
        self.state_ros_time.append(ros_stamp(msg))
        if msg.name:
            self.state_names = list(msg.name)


def as_signal(values, times, ros_times):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        values = np.empty((0, DOF), dtype=np.float64)
    return (
        values,
        np.asarray(times, dtype=np.float64),
        np.asarray(ros_times, dtype=np.float64),
    )


def signal_rate(times):
    if len(times) < 2 or times[-1] <= times[0]:
        return None
    return float((len(times) - 1) / (times[-1] - times[0]))


def jump_summary(values, times, threshold_rad, speed_threshold):
    result = {
        "samples": int(len(values)),
        "rate_hz": signal_rate(times),
        "jump_threshold_rad": float(threshold_rad),
        "speed_threshold_rad_s": float(speed_threshold),
        "jump_count": 0,
        "speed_violation_count": 0,
        "per_joint_max_step_rad": [0.0] * DOF,
        "per_joint_max_speed_rad_s": [0.0] * DOF,
        "events": [],
    }
    if len(values) < 2:
        return result

    delta = np.diff(values, axis=0)
    abs_delta = np.abs(delta)
    dt = np.diff(times)
    speed = np.full_like(abs_delta, np.nan)
    valid = np.isfinite(dt) & (dt > 0.0)
    speed[valid] = abs_delta[valid] / dt[valid, None]
    result["per_joint_max_step_rad"] = np.nanmax(abs_delta, axis=0).tolist()
    result["per_joint_max_speed_rad_s"] = np.nanmax(speed, axis=0).tolist()

    jump_rows, jump_joints = np.where(abs_delta > threshold_rad)
    speed_rows, _ = np.where(speed > speed_threshold)
    result["jump_count"] = int(len(jump_rows))
    result["speed_violation_count"] = int(len(speed_rows))
    for row, joint in zip(jump_rows[:500], jump_joints[:500]):
        result["events"].append(
            {
                "sample": int(row + 1),
                "time": float(times[row + 1]),
                "joint_index": int(joint),
                "joint": JOINT_LABELS[joint],
                "before_rad": float(values[row, joint]),
                "after_rad": float(values[row + 1, joint]),
                "delta_rad": float(delta[row, joint]),
                "speed_rad_s": (
                    float(speed[row, joint])
                    if np.isfinite(speed[row, joint])
                    else None
                ),
            }
        )
    return result


def nearest_indices(reference_time, query_time):
    right = np.searchsorted(reference_time, query_time, side="left")
    right = np.clip(right, 0, len(reference_time) - 1)
    left = np.clip(right - 1, 0, len(reference_time) - 1)
    use_left = np.abs(query_time - reference_time[left]) <= np.abs(
        reference_time[right] - query_time
    )
    return np.where(use_left, left, right)


def tracking_summary(command, command_time, state, state_time, max_lag_ms):
    if len(command) == 0 or len(state) == 0:
        return {
            "samples": 0,
            "estimated_lag_ms": None,
            "rmse_rad": None,
            "max_abs_error_rad": None,
        }

    best = None
    # State(t) is compared with command(t-lag). A small grid is sufficient for
    # diagnostics and avoids pretending command and measured state are identical.
    for lag_ms in np.arange(0.0, max_lag_ms + 0.1, 5.0):
        query = state_time - lag_ms * 1e-3
        indices = nearest_indices(command_time, query)
        error = state - command[indices]
        rmse = float(np.sqrt(np.mean(np.square(error))))
        if best is None or rmse < best[0]:
            best = (rmse, float(lag_ms), error, indices)

    rmse, lag_ms, error, indices = best
    return {
        "samples": int(len(state)),
        "estimated_lag_ms": lag_ms,
        "rmse_rad": rmse,
        "max_abs_error_rad": float(np.max(np.abs(error))),
        "per_joint_rmse_rad": np.sqrt(np.mean(np.square(error), axis=0)).tolist(),
        "per_joint_max_abs_error_rad": np.max(np.abs(error), axis=0).tolist(),
        "note": (
            "joint_states are measured hardware positions, not an echo of the "
            "driver's received target; non-zero tracking error and lag are expected"
        ),
    }


def write_csv(path, values, times, ros_times):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["receive_time", "ros_time"] + list(JOINT_LABELS))
        for timestamp, ros_timestamp, row in zip(times, ros_times, values):
            writer.writerow([timestamp, ros_timestamp] + row.tolist())


def save_recording(recorder, output_dir, jump_rad, speed_rad_s, max_lag_ms):
    output_dir.mkdir(parents=True, exist_ok=True)
    command, command_time, command_ros_time = as_signal(
        recorder.command, recorder.command_time, recorder.command_ros_time
    )
    state, state_time, state_ros_time = as_signal(
        recorder.state, recorder.state_time, recorder.state_ros_time
    )

    arrays = {
        "joint_commands": command,
        "joint_commands_time": command_time,
        "joint_commands_ros_time": command_ros_time,
        "joint_states": state,
        "joint_states_time": state_time,
        "joint_states_ros_time": state_ros_time,
    }
    for name, values in arrays.items():
        np.save(str(output_dir / "{}.npy".format(name)), values)
    write_csv(output_dir / "joint_commands.csv", command, command_time, command_ros_time)
    write_csv(output_dir / "joint_states.csv", state, state_time, state_ros_time)

    report = {
        "command_topic": recorder.command_topic,
        "state_topic": recorder.state_topic,
        "joint_labels": list(JOINT_LABELS),
        "driver_state_names": recorder.state_names,
        "commands": jump_summary(command, command_time, jump_rad, speed_rad_s),
        "states": jump_summary(state, state_time, jump_rad, speed_rad_s),
        "tracking": tracking_summary(
            command, command_time, state, state_time, max_lag_ms
        ),
    }
    with (output_dir / "report.json").open("w") as file:
        json.dump(report, file, indent=2)

    print("\nSaved Wuji signal recording: {}".format(output_dir))
    print(
        "commands: {} samples, {:.1f} Hz, {} jumps".format(
            len(command), report["commands"]["rate_hz"] or 0.0, report["commands"]["jump_count"]
        )
    )
    print(
        "states:   {} samples, {:.1f} Hz, {} jumps".format(
            len(state), report["states"]["rate_hz"] or 0.0, report["states"]["jump_count"]
        )
    )
    tracking = report["tracking"]
    if tracking["rmse_rad"] is not None:
        print(
            "tracking: estimated lag={:.0f} ms, RMSE={:.4f} rad, max error={:.4f} rad".format(
                tracking["estimated_lag_ms"],
                tracking["rmse_rad"],
                tracking["max_abs_error_rad"],
            )
        )
    print("Detailed result: {}".format(output_dir / "report.json"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record /joint_commands and /joint_states for Wuji teleop diagnostics"
    )
    parser.add_argument("--hand-name", default="right_hand")
    parser.add_argument("--output", default=None)
    parser.add_argument("--duration", type=float, default=0.0, help="seconds; 0 means Ctrl+C")
    parser.add_argument("--jump-rad", type=float, default=0.15)
    parser.add_argument("--speed-rad-s", type=float, default=8.0)
    parser.add_argument("--max-lag-ms", type=float, default=500.0)
    parser.add_argument(
        "--print-hz", type=float, default=0.0, help="print live command values at this rate"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hand_name = args.hand_name.strip("/")
    command_topic = "/{}/joint_commands".format(hand_name)
    state_topic = "/{}/joint_states".format(hand_name)
    if args.output:
        output_dir = Path(args.output).expanduser()
    else:
        output_dir = (
            Path.home()
            / "shared_data"
            / "wuji_signal_check"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    rclpy.init()
    recorder = WujiSignalRecorder(command_topic, state_topic, args.print_hz)
    print("Recording {} and {}".format(command_topic, state_topic))
    print("Run teleop now. Press Ctrl+C to stop and save.")
    started = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(recorder, timeout_sec=0.1)
            if args.duration > 0.0 and time.monotonic() - started >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        save_recording(
            recorder,
            output_dir,
            args.jump_rad,
            args.speed_rad_s,
            args.max_lag_ms,
        )
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
