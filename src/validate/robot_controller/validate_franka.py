#!/usr/bin/env python3
"""Validate FrankaController - per-mode left-right oscillation ("도리도리") test.

For each control mode, the robot oscillates joint 1 (base rotation) left/right
around the current position. Data is recorded and saved as .npy for later
inspection.

Usage:
    # 1. Start daemon in Docker:
    docker compose -f docker_robot/docker-compose.yaml exec franka_seoja bash
    /workspace/src_main/cpp_sources/daemon/build/franka_daemon 172.16.1.11 \
        --command_port 5555 --state_port 5556

    # 2. Run all tests:
    python validate_franka.py --host localhost

    # Run specific mode:
    python validate_franka.py --host localhost --mode joint_position

    # Mock test (no daemon):
    python validate_franka.py --mock

    # List available modes:
    python validate_franka.py --list

Saved data structure (per mode):
    <save_dir>/<mode_name>/
        time.npy        (N,)
        qpos.npy        (N, 7)
        qvel.npy        (N, 7)
        wrench.npy      (N, 6)
        O_T_EE.npy      (N, 4, 4)
"""

import argparse
import os
import sys
import time
import datetime
import threading

import numpy as np

# Add paradex to path for FrankaController import
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARADEX_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
sys.path.insert(0, _PARADEX_ROOT)

from paradex.io.robot_controller.franka_controller import FrankaController


# =============================================================================
# Oscillation parameters
# =============================================================================
OSCILLATION_JOINT = 0        # Joint 1 (base rotation)
OSCILLATION_AMPLITUDE = 0.3  # rad (~17.2 deg)
OSCILLATION_CYCLES = 2       # number of left-right cycles
OSCILLATION_SPEED = 0.20     # speed_scale for joint moves

# For velocity / torque modes
VEL_AMPLITUDE = 0.45         # rad/s peak velocity
VEL_STEP_MS = 50             # duration per velocity command (ms)
VEL_OSCILLATION_PERIOD = 2.0 # seconds per full cycle (total: 4s)

TORQUE_AMPLITUDE = 2.0       # Nm peak torque
TORQUE_STEP_MS = 50
TORQUE_OSCILLATION_PERIOD = 2.0  # seconds per full cycle (total: 4s)


# =============================================================================
# Mode tests
# =============================================================================

def test_joint_position(fc: FrankaController, save_dir: str) -> bool:
    """Joint position control: oscillate joint 0 left-right with move_to_qpos."""
    print("\n=== Mode: joint_position (move_to_qpos) ===")

    data = fc.get_data()
    if data is None:
        print("  FAIL: cannot read state")
        return False

    home_qpos = data["qpos"].copy()
    print(f"  Home qpos[0] = {np.degrees(home_qpos[0]):.1f} deg")

    fc.start(save_dir)

    for cycle in range(OSCILLATION_CYCLES):
        # Left
        target = home_qpos.copy()
        target[OSCILLATION_JOINT] = home_qpos[OSCILLATION_JOINT] - OSCILLATION_AMPLITUDE
        print(f"  Cycle {cycle+1}/{OSCILLATION_CYCLES}: LEFT → {np.degrees(target[0]):.1f} deg")
        resp = fc.move(target, speed_scale=OSCILLATION_SPEED)
        if resp.get("type") == "error":
            print(f"  FAIL: {resp.get('message')}")
            fc.stop()
            return False

        # Right
        target[OSCILLATION_JOINT] = home_qpos[OSCILLATION_JOINT] + OSCILLATION_AMPLITUDE
        print(f"  Cycle {cycle+1}/{OSCILLATION_CYCLES}: RIGHT → {np.degrees(target[0]):.1f} deg")
        resp = fc.move(target, speed_scale=OSCILLATION_SPEED)
        if resp.get("type") == "error":
            print(f"  FAIL: {resp.get('message')}")
            fc.stop()
            return False

    # Return home
    print(f"  Returning home...")
    fc.move(home_qpos, speed_scale=OSCILLATION_SPEED)
    fc.stop()

    _print_saved_data(save_dir)
    print("  PASS: joint_position\n")
    return True


def test_cartesian_position(fc: FrankaController, save_dir: str) -> bool:
    """Cartesian position control: oscillate end-effector in Y direction."""
    print("\n=== Mode: cartesian_position (move_to_cartesian) ===")

    data = fc.get_data()
    if data is None:
        print("  FAIL: cannot read state")
        return False

    home_pose = data["position"].copy()  # 4x4
    home_y = home_pose[1, 3]
    print(f"  Home EE position: x={home_pose[0,3]:.3f}, y={home_y:.3f}, z={home_pose[2,3]:.3f}")

    cart_amplitude = 0.03  # 3cm left-right in Y

    fc.start(save_dir)

    for cycle in range(OSCILLATION_CYCLES):
        # Left (−Y)
        target = home_pose.copy()
        target[1, 3] = home_y - cart_amplitude
        print(f"  Cycle {cycle+1}/{OSCILLATION_CYCLES}: LEFT Y={target[1,3]:.3f}")
        resp = fc.move(target, move_speed=0.05)
        if resp.get("type") == "error":
            print(f"  FAIL: {resp.get('message')}")
            fc.stop()
            return False

        # Right (+Y)
        target[1, 3] = home_y + cart_amplitude
        print(f"  Cycle {cycle+1}/{OSCILLATION_CYCLES}: RIGHT Y={target[1,3]:.3f}")
        resp = fc.move(target, move_speed=0.05)
        if resp.get("type") == "error":
            print(f"  FAIL: {resp.get('message')}")
            fc.stop()
            return False

    # Return home
    print(f"  Returning home...")
    fc.move(home_pose, move_speed=0.05)
    fc.stop()

    _print_saved_data(save_dir)
    print("  PASS: cartesian_position\n")
    return True


def test_joint_velocity(fc: FrankaController, save_dir: str) -> bool:
    """Joint velocity control: sinusoidal oscillation on joint 0 (streaming mode)."""
    print("\n=== Mode: joint_velocity (set_joint_velocity, streaming) ===")

    data = fc.get_data()
    if data is None:
        print("  FAIL: cannot read state")
        return False

    print(f"  Initial qpos[0] = {np.degrees(data['qpos'][0]):.1f} deg")
    print(f"  Will oscillate with peak vel = {np.degrees(VEL_AMPLITUDE):.1f} deg/s")

    fc.start(save_dir)

    t0 = time.time()
    duration = VEL_OSCILLATION_PERIOD * OSCILLATION_CYCLES
    step_interval = VEL_STEP_MS / 1000.0

    while time.time() - t0 < duration:
        t = time.time() - t0
        # Sinusoidal velocity: v(t) = A * sin(2π * t / T)
        vel = VEL_AMPLITUDE * np.sin(2.0 * np.pi * t / VEL_OSCILLATION_PERIOD)
        dq = np.zeros(7)
        dq[OSCILLATION_JOINT] = vel

        resp = fc.set_joint_velocity(dq, duration_ms=0)  # streaming mode
        if resp.get("type") == "error":
            print(f"  FAIL: {resp.get('message')}")
            fc.stop_streaming()
            fc.stop()
            return False

        time.sleep(step_interval)

    # Stop streaming (daemon will ramp to zero smoothly)
    fc.stop_streaming()
    fc.stop()

    _print_saved_data(save_dir)
    print("  PASS: joint_velocity\n")
    return True


def test_cartesian_velocity(fc: FrankaController, save_dir: str) -> bool:
    """Cartesian velocity control: sinusoidal oscillation in Y (streaming mode)."""
    print("\n=== Mode: cartesian_velocity (set_cartesian_velocity, streaming) ===")

    data = fc.get_data()
    if data is None:
        print("  FAIL: cannot read state")
        return False

    cart_vel_amp = 0.02  # m/s in Y direction
    print(f"  Will oscillate EE in Y with peak vel = {cart_vel_amp*100:.1f} cm/s")

    fc.start(save_dir)

    t0 = time.time()
    duration = VEL_OSCILLATION_PERIOD * OSCILLATION_CYCLES
    step_interval = VEL_STEP_MS / 1000.0

    while time.time() - t0 < duration:
        t = time.time() - t0
        vy = cart_vel_amp * np.sin(2.0 * np.pi * t / VEL_OSCILLATION_PERIOD)
        twist = [0, vy, 0, 0, 0, 0]

        resp = fc.set_cartesian_velocity(twist, duration_ms=0)
        if resp.get("type") == "error":
            print(f"  FAIL: {resp.get('message')}")
            fc.stop_streaming()
            fc.stop()
            return False

        time.sleep(step_interval)

    fc.stop_streaming()
    fc.stop()

    _print_saved_data(save_dir)
    print("  PASS: cartesian_velocity\n")
    return True


def test_torque(fc: FrankaController, save_dir: str) -> bool:
    """Direct torque control: gentle sinusoidal torque on joint 4 (wrist, streaming mode)."""
    TORQUE_JOINT = 4  # wrist joint — light, moves easily with small torque
    print("\n=== Mode: torque (set_torques, streaming) ===")
    print(f"  WARNING: Direct torque control. Joint {TORQUE_JOINT}, peak = {TORQUE_AMPLITUDE} Nm")

    data = fc.get_data()
    if data is None:
        print("  FAIL: cannot read state")
        return False

    # Countdown so user can walk to the robot
    for i in range(10, 0, -1):
        print(f"  Starting in {i}...", flush=True)
        time.sleep(1)
    print("  GO!")

    fc.start(save_dir)

    # Single streaming session — just change target values between phases
    step_interval = TORQUE_STEP_MS / 1000.0

    # --- Phase 1: Sinusoidal oscillation (8 sec) ---
    print(f"\n  Phase 1: Sinusoidal torque on joint {TORQUE_JOINT}, {TORQUE_AMPLITUDE} Nm peak (8 sec)")
    t0 = time.time()
    duration = TORQUE_OSCILLATION_PERIOD * 4  # 4 cycles = 8 sec
    while time.time() - t0 < duration:
        t = time.time() - t0
        tau = TORQUE_AMPLITUDE * np.sin(2.0 * np.pi * t / TORQUE_OSCILLATION_PERIOD)
        torques = np.zeros(7)
        torques[TORQUE_JOINT] = tau
        resp = fc.set_torques(torques, duration_ms=0)
        if resp.get("type") == "error":
            print(f"  FAIL: {resp.get('message')}")
            fc.stop_streaming()
            return False
        time.sleep(step_interval)

    # --- Phase 2: Zero torque (10 sec) ---
    print("\n  Phase 2: ZERO TORQUE — push the robot, it should move freely (10 sec)")
    t0 = time.time()
    while time.time() - t0 < 10.0:
        resp = fc.set_torques(np.zeros(7).tolist(), duration_ms=0)
        if resp.get("type") == "error":
            print(f"  FAIL: {resp.get('message')}")
            fc.stop_streaming()
            return False
        time.sleep(0.05)

    # --- Phase 3: Constant torque (10 sec) ---
    hold_torque = 1.5  # Nm
    print(f"\n  Phase 3: CONSTANT {hold_torque} Nm on joint {TORQUE_JOINT} — push against it (10 sec)")
    t0 = time.time()
    while time.time() - t0 < 10.0:
        torques = np.zeros(7)
        torques[TORQUE_JOINT] = hold_torque
        resp = fc.set_torques(torques, duration_ms=0)
        if resp.get("type") == "error":
            print(f"  FAIL: {resp.get('message')}")
            fc.stop_streaming()
            return False
        time.sleep(0.05)

    # Stop streaming once at the end
    fc.stop_streaming()
    fc.stop()

    _print_saved_data(save_dir)
    print("  PASS: torque\n")
    return True


def test_impedance_config(fc: FrankaController, save_dir: str) -> bool:
    """Test impedance configuration: set soft/stiff, then hold position while user pushes.

    The robot holds position. User pushes the robot to feel the difference:
    - SOFT: robot deflects easily (feels springy/compliant)
    - STIFF: robot resists pushing (feels rigid)
    """
    print("\n=== Mode: impedance_config ===")
    print("  This test lets you PUSH the robot to feel impedance difference.")

    data = fc.get_data()
    if data is None:
        print("  FAIL: cannot read state")
        return False

    home_qpos = data["qpos"].copy()

    fc.start(save_dir)

    # --- Phase 1: SOFT impedance ---
    print("\n  Setting SOFT joint impedance: [200, 200, 200, 200, 100, 50, 20]")
    resp = fc.set_joint_impedance([200, 200, 200, 200, 100, 50, 20])
    if resp.get("type") == "error":
        print(f"  FAIL: {resp.get('message')}")
        fc.stop()
        return False
    print("  [OK] set_joint_impedance (soft)")

    # Oscillation so user can see motion
    for cycle in range(2):
        target = home_qpos.copy()
        target[OSCILLATION_JOINT] -= OSCILLATION_AMPLITUDE * 0.5
        fc.move(target, speed_scale=OSCILLATION_SPEED)
        target[OSCILLATION_JOINT] += OSCILLATION_AMPLITUDE
        fc.move(target, speed_scale=OSCILLATION_SPEED)
    fc.move(home_qpos, speed_scale=OSCILLATION_SPEED)

    # Hold position — user pushes robot to feel softness
    print("  >>> SOFT: Push the robot now! It should feel compliant. (5 sec) <<<")
    t0 = time.time()
    while time.time() - t0 < 5.0:
        fc.move(home_qpos, speed_scale=OSCILLATION_SPEED)
        time.sleep(0.1)

    # --- Phase 2: STIFF impedance ---
    print("\n  Setting STIFF joint impedance: [600, 600, 600, 600, 250, 150, 50]")
    resp = fc.set_joint_impedance([600, 600, 600, 600, 250, 150, 50])
    if resp.get("type") == "error":
        print(f"  FAIL: {resp.get('message')}")
        fc.stop()
        return False
    print("  [OK] set_joint_impedance (stiff)")

    # Oscillation
    for cycle in range(2):
        target = home_qpos.copy()
        target[OSCILLATION_JOINT] -= OSCILLATION_AMPLITUDE * 0.5
        fc.move(target, speed_scale=OSCILLATION_SPEED)
        target[OSCILLATION_JOINT] += OSCILLATION_AMPLITUDE
        fc.move(target, speed_scale=OSCILLATION_SPEED)
    fc.move(home_qpos, speed_scale=OSCILLATION_SPEED)

    # Hold position — user pushes robot to feel stiffness
    print("  >>> STIFF: Push the robot now! It should feel rigid. (5 sec) <<<")
    t0 = time.time()
    while time.time() - t0 < 5.0:
        fc.move(home_qpos, speed_scale=OSCILLATION_SPEED)
        time.sleep(0.1)

    fc.stop()

    # Test Cartesian impedance config (just verify command works)
    print("\n  Setting Cartesian impedance: [3000, 3000, 3000, 300, 300, 300]")
    resp = fc.set_cartesian_impedance([3000, 3000, 3000, 300, 300, 300])
    if resp.get("type") == "error":
        print(f"  FAIL: {resp.get('message')}")
        return False
    print("  [OK] set_cartesian_impedance")

    _print_saved_data(save_dir)
    print("  PASS: impedance_config\n")
    return True


def test_collision_behavior(fc: FrankaController, save_dir: str) -> bool:
    """Test collision behavior thresholds + oscillation."""
    print("\n=== Mode: collision_behavior ===")

    data = fc.get_data()
    if data is None:
        print("  FAIL: cannot read state")
        return False

    home_qpos = data["qpos"].copy()

    # Set relaxed collision thresholds
    print("  Setting relaxed collision thresholds...")
    resp = fc.set_collision_behavior(
        torque_lower=[40.0] * 7,
        torque_upper=[50.0] * 7,
        force_lower=[40.0] * 6,
        force_upper=[50.0] * 6,
    )
    if resp.get("type") == "error":
        print(f"  FAIL: {resp.get('message')}")
        return False
    print("  [OK] set_collision_behavior")

    # Oscillate to verify motion works under new thresholds
    fc.start(save_dir)
    for cycle in range(OSCILLATION_CYCLES):
        target = home_qpos.copy()
        target[OSCILLATION_JOINT] -= OSCILLATION_AMPLITUDE
        fc.move(target, speed_scale=OSCILLATION_SPEED)
        target[OSCILLATION_JOINT] += OSCILLATION_AMPLITUDE * 2
        fc.move(target, speed_scale=OSCILLATION_SPEED)

    fc.move(home_qpos, speed_scale=OSCILLATION_SPEED)
    fc.stop()

    _print_saved_data(save_dir)
    print("  PASS: collision_behavior\n")
    return True


def test_set_load(fc: FrankaController, save_dir: str) -> bool:
    """Test setLoad (payload registration) + oscillation."""
    print("\n=== Mode: set_load ===")

    data = fc.get_data()
    if data is None:
        print("  FAIL: cannot read state")
        return False

    home_qpos = data["qpos"].copy()

    # Register a 0.5 kg payload
    print("  Setting load: 0.5 kg at [0, 0, 0.04]m...")
    resp = fc.set_load(
        mass=0.5,
        F_x_Cload=[0.0, 0.0, 0.04],
        load_inertia=np.diag([0.001, 0.001, 0.001]),
    )
    if resp.get("type") == "error":
        print(f"  FAIL: {resp.get('message')}")
        return False
    print("  [OK] set_load")

    # Oscillate under declared load
    fc.start(save_dir)
    for cycle in range(OSCILLATION_CYCLES):
        target = home_qpos.copy()
        target[OSCILLATION_JOINT] -= OSCILLATION_AMPLITUDE
        fc.move(target, speed_scale=OSCILLATION_SPEED)
        target[OSCILLATION_JOINT] += OSCILLATION_AMPLITUDE * 2
        fc.move(target, speed_scale=OSCILLATION_SPEED)

    fc.move(home_qpos, speed_scale=OSCILLATION_SPEED)
    fc.stop()

    # Reset load to zero
    print("  Resetting load to zero...")
    fc.set_load(mass=0.0, F_x_Cload=[0, 0, 0], load_inertia=np.zeros((3, 3)))

    _print_saved_data(save_dir)
    print("  PASS: set_load\n")
    return True


def test_error_recovery(fc: FrankaController, save_dir: str) -> bool:
    """Test error recovery command."""
    print("\n=== Mode: error_recovery ===")

    resp = fc.error_recovery()
    if resp.get("type") == "error":
        print(f"  INFO: error_recovery returned error (may be normal if no error): {resp.get('message')}")
    else:
        print("  [OK] error_recovery")

    print("  PASS: error_recovery\n")
    return True


def test_gripper(fc: FrankaController, save_dir: str) -> bool:
    """Test gripper open/close cycle."""
    print("\n=== Mode: gripper ===")

    fc.start(save_dir)

    for cycle in range(2):
        print(f"  Cycle {cycle+1}/2: Opening...")
        resp = fc.open_gripper(width=0.08, speed=0.05)
        if resp.get("type") == "error":
            print(f"  FAIL: open_gripper: {resp.get('message')}")
            fc.stop()
            return False
        time.sleep(0.5)

        data = fc.get_data()
        print(f"  Gripper width: {data['gripper_width']:.4f}m")

        print(f"  Cycle {cycle+1}/2: Grasping...")
        resp = fc.grasp(force=20.0, speed=0.05)
        if resp.get("type") == "error":
            print(f"  FAIL: grasp: {resp.get('message')}")
            fc.stop()
            return False
        time.sleep(0.5)

        data = fc.get_data()
        print(f"  Gripper width: {data['gripper_width']:.4f}m, grasping: {data['gripper_grasping']}")

    # End open
    fc.open_gripper(width=0.08, speed=0.05)
    fc.stop()

    _print_saved_data(save_dir)
    print("  PASS: gripper\n")
    return True


# =============================================================================
# Mock test (no daemon)
# =============================================================================

def test_mock():
    """Quick serialization and socket test without daemon."""
    import zmq
    import msgpack

    print("=== Mock Test (no daemon needed) ===")

    commands = [
        {"type": "ping"},
        {"type": "get_state"},
        {"type": "move_to_qpos", "qpos": [0.0] * 7, "speed_scale": 0.15},
        {"type": "move_to_cartesian", "position": [0.3, 0.0, 0.5],
         "orientation": [1, 0, 0, 0, 1, 0, 0, 0, 1], "move_speed": 0.05},
        {"type": "set_joint_velocity", "dq": [0.0] * 7, "duration_ms": 100},
        {"type": "set_cartesian_velocity", "twist": [0.0] * 6, "duration_ms": 100},
        {"type": "set_torques", "torques": [0.0] * 7, "duration_ms": 100},
        {"type": "set_joint_impedance", "K_theta": [600.0] * 7},
        {"type": "set_cartesian_impedance", "K_x": [3000, 3000, 3000, 300, 300, 300]},
        {"type": "set_collision_behavior",
         "torque_lower": [20.0] * 7, "torque_upper": [20.0] * 7,
         "force_lower": [20.0] * 6, "force_upper": [20.0] * 6},
        {"type": "set_load", "mass": 0.5, "F_x_Cload": [0, 0, 0.04],
         "load_inertia": [0.001, 0, 0, 0, 0.001, 0, 0, 0, 0.001]},
        {"type": "set_ee", "NE_T_EE": list(range(16))},
        {"type": "set_guiding_mode", "guiding_axes": [True] * 6, "nullspace": False},
        {"type": "error_recovery"},
        {"type": "open_gripper", "width": 0.08, "speed": 0.05},
        {"type": "grasp", "force": 60.0, "speed": 0.05, "eps_inner": 0.005, "eps_outer": 0.005},
        {"type": "stop"},
    ]

    for cmd in commands:
        packed = msgpack.packb(cmd, use_bin_type=True)
        unpacked = msgpack.unpackb(packed, raw=False)
        assert unpacked == cmd, f"Roundtrip failed for {cmd['type']}"
        print(f"  [OK] {cmd['type']} ({len(packed)} bytes)")

    # ZMQ socket lifecycle
    ctx = zmq.Context()
    req = ctx.socket(zmq.REQ)
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    req.close()
    sub.close()
    ctx.term()
    print("  [OK] ZMQ socket create/close")

    print("=== Mock Test PASSED ===\n")


# =============================================================================
# Helpers
# =============================================================================

def _print_saved_data(save_dir: str):
    """Print summary of saved npy files."""
    if not os.path.isdir(save_dir):
        print(f"  (no data saved)")
        return
    for f in sorted(os.listdir(save_dir)):
        if f.endswith(".npy"):
            arr = np.load(os.path.join(save_dir, f))
            print(f"  Saved: {f} shape={arr.shape}")


# =============================================================================
# Mode registry
# =============================================================================

MODE_TESTS = {
    "joint_position": test_joint_position,
    "cartesian_position": test_cartesian_position,
    "joint_velocity": test_joint_velocity,
    "cartesian_velocity": test_cartesian_velocity,
    "torque": test_torque,
    "impedance_config": test_impedance_config,
    "collision_behavior": test_collision_behavior,
    "set_load": test_set_load,
    "error_recovery": test_error_recovery,
    "gripper": test_gripper,
}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate FrankaController - per-mode oscillation test"
    )
    parser.add_argument("--host", type=str, default=None, help="Daemon host (e.g. localhost)")
    parser.add_argument("--command_port", type=int, default=5555)
    parser.add_argument("--state_port", type=int, default=5556)
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all"] + list(MODE_TESTS.keys()),
        help="Which mode to test (default: all)",
    )
    parser.add_argument("--mock", action="store_true", help="Run mock test only (no daemon)")
    parser.add_argument("--list", action="store_true", help="List available modes")
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="Directory for saving recorded data (default: auto-timestamped)",
    )
    args = parser.parse_args()

    if args.list:
        print("Available modes:")
        for name in MODE_TESTS:
            print(f"  - {name}")
        return 0

    if args.mock or args.host is None:
        test_mock()
        print("All mock tests passed!")
        return 0

    # Create save directory
    if args.save_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = os.path.join(
            _THIS_DIR, "..", "..", "..", "data", "validate_franka", ts
        )
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Save directory: {os.path.abspath(args.save_dir)}")

    # Connect
    print(f"\nConnecting to daemon at {args.host}:{args.command_port}...")
    fc = FrankaController(args.host, args.command_port, args.state_port)

    if not fc.ping():
        print("FAIL: ping failed")
        return 1
    print("[OK] Connected\n")

    # Safety prompt
    print("=" * 60)
    print("  ROBOT WILL MOVE! Ensure workspace is clear.")
    print("  Each mode oscillates the robot left-right (~5 deg).")
    print("=" * 60)
    input("Press Enter to start or Ctrl+C to abort... ")

    # Run tests
    if args.mode == "all":
        modes_to_test = list(MODE_TESTS.keys())
    else:
        modes_to_test = [args.mode]

    results = {}
    for mode_name in modes_to_test:
        mode_save = os.path.join(args.save_dir, mode_name)
        os.makedirs(mode_save, exist_ok=True)

        # Error recovery before each test to clear leftover error state
        try:
            fc.error_recovery()
        except Exception:
            pass
        time.sleep(0.3)

        try:
            passed = MODE_TESTS[mode_name](fc, mode_save)
        except KeyboardInterrupt:
            print(f"\n  Interrupted during {mode_name}")
            fc.emergency_stop()
            break
        except Exception as e:
            print(f"  EXCEPTION in {mode_name}: {e}")
            passed = False
            try:
                fc.stop_streaming()
            except Exception:
                pass

        results[mode_name] = passed

    fc.end()

    # Summary
    print("\n" + "=" * 40)
    print("  SUMMARY")
    print("=" * 40)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    print(f"\n  Data saved to: {os.path.abspath(args.save_dir)}")
    print("=" * 40)

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
