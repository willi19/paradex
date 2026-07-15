"""Validate Quest 3 wrist teleoperation with xArm only.

This bypasses CaptureSession, cameras, recording, and hand controllers. The
Quest app must send the existing HandStream JSON packets to this machine.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paradex.io.teleop.occulus.receiver import Quest3Receiver
from paradex.retargetor.unimanual import Retargetor


class QuestXArmTeleopController:
    """Apply ParaDex relative-motion and pause/rebase semantics to wrist poses."""

    def __init__(
        self,
        home_pose: np.ndarray,
        max_offset_m: float = 0.30,
        max_step_m: float = 0.05,
    ) -> None:
        self.home_pose = np.asarray(home_pose, dtype=np.float64).copy()
        self.max_offset_m = max(0.0, float(max_offset_m))
        self.max_step_m = max(0.0, float(max_step_m))
        self.retargetor = Retargetor(
            arm_name="xarm", hand_name=None, hand_side="Right"
        )
        self.retargetor.start(self.home_pose)
        self.last_command = self.home_pose.copy()
        self.paused = False
        self.tracking = False

    @staticmethod
    def _limit_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
        if max_norm <= 0.0:
            return vector
        norm = np.linalg.norm(vector)
        if norm <= max_norm or norm < 1e-12:
            return vector
        return vector * (max_norm / norm)

    def _bounded_target(self, target: np.ndarray) -> np.ndarray:
        bounded = target.copy()
        home_offset = bounded[:3, 3] - self.home_pose[:3, 3]
        bounded[:3, 3] = self.home_pose[:3, 3] + self._limit_vector(
            home_offset, self.max_offset_m
        )
        command_step = bounded[:3, 3] - self.last_command[:3, 3]
        bounded[:3, 3] = self.last_command[:3, 3] + self._limit_vector(
            command_step, self.max_step_m
        )
        return bounded

    def update(
        self, data: Dict, control_state: int
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        right_hand = data.get("Right")
        if right_hand is None:
            if self.tracking:
                self.retargetor.stop()
                self.tracking = False
                return None, "tracking_lost"
            return None, None

        event = None
        if not self.tracking:
            self.retargetor.stop()
            self.tracking = True
            event = "tracking_acquired"

        next_paused = int(control_state) != 0
        if next_paused:
            if not self.paused:
                self.retargetor.stop()
                event = "paused"
            self.paused = True
            return None, event

        if self.paused:
            self.retargetor.stop()
            event = "resumed"
        self.paused = False

        target, _ = self.retargetor.get_action(data)
        target = self._bounded_target(target)
        self.last_command = target.copy()
        self.retargetor.last_arm_pose["Right"] = target.copy()
        return target, event


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quest 3 wrist-only teleoperation validator for xArm"
    )
    parser.add_argument("--quest-bind-host", default="0.0.0.0")
    parser.add_argument("--quest-port", type=int, default=9000)
    parser.add_argument(
        "--quest-max-age-s",
        type=float,
        default=0.75,
        help="Hold the last safe Quest pose for this long before tracking loss.",
    )
    parser.add_argument("--quest-extrapolation-delay-s", type=float, default=0.03)
    parser.add_argument("--quest-extrapolation-horizon-s", type=float, default=0.10)
    parser.add_argument("--quest-max-linear-speed-m-s", type=float, default=0.5)
    parser.add_argument("--quest-max-angular-speed-rad-s", type=float, default=3.0)
    parser.add_argument("--wait-timeout-s", type=float, default=15.0)
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--max-offset-m", type=float, default=0.60)
    parser.add_argument("--max-step-m", type=float, default=0.10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ignore-arm-errors", action="store_true")
    return parser


def wait_for_right_hand(receiver: Quest3Receiver, timeout_s: float) -> Dict:
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while time.monotonic() < deadline:
        data = receiver.get_data()
        if data.get("Right") is not None:
            return data
        time.sleep(0.01)
    raise TimeoutError(
        "No fresh Quest right-hand packet received on UDP port {}".format(
            receiver.port
        )
    )


def run(args: argparse.Namespace) -> int:
    receiver = Quest3Receiver(
        host=args.quest_bind_host,
        port=args.quest_port,
        max_age_s=args.quest_max_age_s,
        extrapolation_delay_s=args.quest_extrapolation_delay_s,
        extrapolation_horizon_s=args.quest_extrapolation_horizon_s,
        max_linear_speed_m_s=args.quest_max_linear_speed_m_s,
        max_angular_speed_rad_s=args.quest_max_angular_speed_rad_s,
    )
    arm = None
    try:
        print(
            "Waiting for Quest right-hand packets on UDP {}:{}...".format(
                args.quest_bind_host, receiver.port
            )
        )
        initial_data = wait_for_right_hand(receiver, args.wait_timeout_s)

        if args.dry_run:
            home_pose = np.eye(4, dtype=np.float64)
            print("Dry-run enabled: xArm will not be initialized or commanded.")
        else:
            from paradex.io.robot_controller import get_arm

            print("Initializing xArm ROS controller...")
            arm = get_arm("xarm", servo_api="cartesian_aa")
            home_pose = np.asarray(arm.get_data()["position"], dtype=np.float64)
            if home_pose.shape != (4, 4) or not np.all(np.isfinite(home_pose)):
                raise RuntimeError("xArm did not return a valid 4x4 TCP pose")

        controller = QuestXArmTeleopController(
            home_pose=home_pose,
            max_offset_m=args.max_offset_m,
            max_step_m=args.max_step_m,
        )
        controller.update(initial_data, receiver.get_control_state())

        print("Teleoperation ready: left hand open=pause, fist=resume/rebase.")
        print("Press Ctrl+C to stop.")
        period_s = 1.0 / max(1.0, float(args.rate_hz))
        deadline = (
            time.monotonic() + args.duration_s if args.duration_s > 0.0 else None
        )
        last_status_time = 0.0
        target = home_pose.copy()

        while deadline is None or time.monotonic() < deadline:
            loop_start = time.monotonic()
            data = receiver.get_data()
            control_state = receiver.get_control_state()
            next_target, event = controller.update(data, control_state)

            if event is not None:
                print("teleop event: {}".format(event))
            if next_target is not None:
                target = next_target
                if arm is not None:
                    arm.move(target)

            if arm is not None and arm.is_error() and not args.ignore_arm_errors:
                raise RuntimeError("xArm controller reported a command/service error")

            now = time.monotonic()
            if now - last_status_time >= 1.0:
                last_status_time = now
                print(
                    "status tracking={} paused={} control_state={} target_xyz_m={}".format(
                        controller.tracking,
                        controller.paused,
                        control_state,
                        np.array2string(target[:3, 3], precision=3),
                    )
                )

            elapsed = time.monotonic() - loop_start
            if elapsed < period_s:
                time.sleep(period_s - elapsed)

    except KeyboardInterrupt:
        print("Stopped by user.")
    except (RuntimeError, TimeoutError, OSError) as exc:
        print("ERROR: {}".format(exc))
        return 1
    finally:
        receiver.end()
        if arm is not None:
            arm.end()
    return 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
