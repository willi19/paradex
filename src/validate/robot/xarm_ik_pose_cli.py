"""
Run xArm pose command and inspect result.

Examples
1) One-shot (IK + set_servo_angle_j):
   python src/validate/robot/xarm_ik_pose_cli.py --pose "300 0 250 0 0 0" --hold-sec 0.2

2) One-shot (set_servo_cartesian_aa):
   python src/validate/robot/xarm_ik_pose_cli.py --method cartesian_aa --pose "300 0 250 0 0 0"

3) Sequential compare (requested flow):
   python src/validate/robot/xarm_ik_pose_cli.py --method compare --pose "300 0 250 0 0 0"

4) Interactive mode:
   python src/validate/robot/xarm_ik_pose_cli.py
   then type: 300 0 250 0 0 0

Pose format (6D axis-angle):
- x y z: millimeters
- ax ay az: axis-angle vector (degrees, 입력)
"""

import argparse
import sys
import time
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation
import transforms3d as t3d

from paradex.io.robot_controller import get_arm


PRINT_PRECISION = 6


def _arr_str(x):
    return np.array2string(np.asarray(x), precision=PRINT_PRECISION)


def parse_pose_tokens(tokens):
    if len(tokens) != 6:
        raise ValueError(f"Expected 6 numbers, got {len(tokens)}")
    pose = np.asarray([float(v) for v in tokens], dtype=np.float64)
    # User input convention: xyz[mm], axis-angle[deg]
    pose[3:6] = np.deg2rad(pose[3:6])
    return pose


def homo2aa(h):
    t = h[:3, 3] * 1000.0
    axis, angle = t3d.axangles.mat2axangle(h[:3, :3], unit_thresh=0.001)
    return np.concatenate([t, axis * angle])


def aa2homo(aa):
    pos = np.eye(4)
    pos[:3, 3] = np.asarray(aa[:3], dtype=np.float64) / 1000.0
    axis_angle = np.asarray(aa[3:6], dtype=np.float64)
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-12:
        pos[:3, :3] = np.eye(3)
    else:
        axis = axis_angle / angle
        pos[:3, :3] = t3d.axangles.axangle2mat(axis, angle)
    return pos


def pose_error(target_h, measured_h):
    pos_err_mm = np.linalg.norm((measured_h[:3, 3] - target_h[:3, 3]) * 1000.0)
    dR = target_h[:3, :3].T @ measured_h[:3, :3]
    rot_err_rad = np.linalg.norm(Rotation.from_matrix(dR).as_rotvec())
    return float(pos_err_mm), float(rot_err_rad)


def print_result(tag, data):
    qpos = data.get("qpos", np.full(6, np.nan))
    pos_h = data.get("position", np.eye(4))
    pose_aa_rad = homo2aa(np.asarray(pos_h))
    pose_aa_deg = pose_aa_rad.copy()
    pose_aa_deg[3:6] = np.rad2deg(pose_aa_deg[3:6])
    print(f"[{tag}] qpos(rad): {_arr_str(qpos)}")
    print(f"[{tag}] pose_aa(mm/deg): {_arr_str(pose_aa_deg)}")


def run_once(arm, pose_aa, hold_sec):
    arm.move(pose_aa)
    if hold_sec > 0.0:
        time.sleep(hold_sec)
    data = arm.get_data()
    print_result("RESULT", data)


def interactive_loop(arm, hold_sec):
    print("Interactive mode. Enter 6D pose: x y z ax ay az (mm, deg)")
    print("Type 'q' to quit.")
    while True:
        try:
            line = input("pose> ").strip()
        except EOFError:
            print("\nEOF. Exit.")
            return

        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            return

        try:
            pose = parse_pose_tokens(line.replace(",", " ").split())
        except Exception as e:
            print(f"Invalid input: {e}")
            continue

        run_once(arm, pose, hold_sec)


def build_arm(method, ee_frame, fps):
    if method == "ik_angle_j":
        return get_arm("xarm_ik", ee_frame=ee_frame, fps=fps)
    if method == "cartesian_aa":
        return get_arm("xarm", servo_api="cartesian_aa", fps=fps)
    raise ValueError(f"Unsupported method: {method}")


def _wait_valid_qpos(arm, timeout_sec=3.0):
    deadline = time.time() + timeout_sec
    last = None
    while time.time() < deadline:
        last = arm.get_data()
        q = np.asarray(last.get("qpos", np.full(6, np.nan)), dtype=np.float64)
        if q.shape == (6,) and np.all(np.isfinite(q)):
            return q, last
        time.sleep(0.02)
    raise RuntimeError(f"Failed to read valid qpos within {timeout_sec:.1f}s (last={last})")


def run_sequential_compare(pose_aa, hold_sec, ee_frame, fps):
    target_h = aa2homo(pose_aa)
    target_pose_deg = np.asarray(pose_aa, dtype=np.float64).copy()
    target_pose_deg[3:6] = np.rad2deg(target_pose_deg[3:6])

    arm_ik = None
    arm_angle = None
    arm_cart = None
    try:
        # 1) Save current joint pose
        arm_ik = get_arm("xarm_ik", ee_frame=ee_frame, fps=fps)
        fk_solver = arm_ik.ik_solver
        time.sleep(0.2)
        init_qpos, _ = _wait_valid_qpos(arm_ik)
        init_data = arm_ik.get_data()
        init_pose_aa = homo2aa(np.asarray(init_data["position"], dtype=np.float64))
        init_pose_aa_deg = init_pose_aa.copy()
        init_pose_aa_deg[3:6] = np.rad2deg(init_pose_aa_deg[3:6])
        print(f"[STEP1] initial_qpos(rad): {_arr_str(init_qpos)}")
        print(f"[STEP1] initial_pose_aa(mm/deg): {_arr_str(init_pose_aa_deg)}")

        # 2,3) input pose -> IK solve + set_angle_j -> error
        t0 = time.perf_counter()
        arm_ik.move(pose_aa)
        if hold_sec > 0.0:
            time.sleep(hold_sec)
        data_ik = arm_ik.get_data()
        dt_ik_ms = (time.perf_counter() - t0) * 1000.0
        qpos_after_angle = np.asarray(data_ik.get("qpos", np.full(6, np.nan)), dtype=np.float64)
        pose_after_angle_h = np.asarray(data_ik["position"], dtype=np.float64)
        pose_after_angle_aa = homo2aa(pose_after_angle_h)
        pose_after_angle_deg = pose_after_angle_aa.copy()
        pose_after_angle_deg[3:6] = np.rad2deg(pose_after_angle_deg[3:6])
        err_ik_mm, err_ik_rad = pose_error(target_h, pose_after_angle_h)

        # IK solution quality in URDF FK space (independent from real robot tracking error)
        ik_action_qpos = np.full(6, np.nan, dtype=np.float64)
        ik_fk_pose_aa = np.full(6, np.nan, dtype=np.float64)
        ik_fk_err_mm = np.nan
        ik_fk_err_rad = np.nan
        try:
            ik_q = np.asarray(getattr(arm_ik, "last_cmd_qpos"), dtype=np.float64).copy()
            ik_action_qpos = ik_q.copy()
            ik_fk_h = arm_ik.ik_solver.fk(ik_q)
            ik_fk_pose_aa = homo2aa(ik_fk_h)
            ik_fk_err_mm, ik_fk_err_rad = pose_error(target_h, ik_fk_h)
        except Exception:
            pass
        ik_fk_pose_deg = ik_fk_pose_aa.copy()
        ik_fk_pose_deg[3:6] = np.rad2deg(ik_fk_pose_deg[3:6])

        print(f"[STEP3] IK+set_angle_j done ({dt_ik_ms:.3f} ms)")
        arm_ik.end()
        arm_ik = None

        # 4) return to initial with set_angle_j
        arm_angle = get_arm("xarm", servo_api="angle_j", fps=fps)
        time.sleep(0.2)
        arm_angle.move(init_qpos)
        if hold_sec > 0.0:
            time.sleep(hold_sec)
        _ = arm_angle.get_data()
        print("[STEP4] returned to initial pose by set_angle_j")
        arm_angle.end()
        arm_angle = None

        # 5,6) move by set_servo_cartesian_aa -> error
        arm_cart = get_arm("xarm", servo_api="cartesian_aa", fps=fps)
        time.sleep(0.2)
        t1 = time.perf_counter()
        arm_cart.move(pose_aa)
        if hold_sec > 0.0:
            time.sleep(hold_sec)
        data_cart = arm_cart.get_data()
        dt_cart_ms = (time.perf_counter() - t1) * 1000.0
        qpos_after_servo = np.asarray(data_cart.get("qpos", np.full(6, np.nan)), dtype=np.float64)
        pose_after_servo_h = np.asarray(data_cart["position"], dtype=np.float64)
        pose_after_servo_aa = homo2aa(pose_after_servo_h)
        pose_after_servo_deg = pose_after_servo_aa.copy()
        pose_after_servo_deg[3:6] = np.rad2deg(pose_after_servo_deg[3:6])
        err_cart_mm, err_cart_rad = pose_error(target_h, pose_after_servo_h)
        print(f"[STEP6] set_servo_cartesian_aa done ({dt_cart_ms:.3f} ms)")

        # URDF FK from observed joint state after set_servo_cartesian_aa
        servo_fk_pose_aa = np.full(6, np.nan, dtype=np.float64)
        servo_fk_err_mm = np.nan
        servo_fk_err_rad = np.nan
        try:
            if np.all(np.isfinite(qpos_after_servo)):
                servo_fk_h = fk_solver.fk(qpos_after_servo)
                servo_fk_pose_aa = homo2aa(servo_fk_h)
                servo_fk_err_mm, servo_fk_err_rad = pose_error(target_h, servo_fk_h)
        except Exception:
            pass
        servo_fk_pose_deg = servo_fk_pose_aa.copy()
        servo_fk_pose_deg[3:6] = np.rad2deg(servo_fk_pose_deg[3:6])

        # 7) return to initial 6D pose by set_servo_cartesian_aa
        arm_cart.move(init_pose_aa)
        if hold_sec > 0.0:
            time.sleep(hold_sec)
        data_back = arm_cart.get_data()
        pose_back_aa = homo2aa(np.asarray(data_back["position"], dtype=np.float64))
        print("[STEP7] returned to initial 6D pose by set_servo_cartesian_aa")

        print("\n=== Compare Result ===")
        print(f"target_pose_aa(mm/deg):  {_arr_str(target_pose_deg)}")
        print(f"initial_pose_aa(mm/deg): {_arr_str(init_pose_aa_deg)}")
        print(f"ik_action_qpos(rad):     {_arr_str(ik_action_qpos)}")
        print(f"ik_fk_pose_aa(mm/deg):   {_arr_str(ik_fk_pose_deg)}")
        print(f"after_angle_qpos(rad):   {_arr_str(qpos_after_angle)}")
        print(f"after_angle_aa(mm/deg):  {_arr_str(pose_after_angle_deg)}")
        print(f"after_servo_qpos(rad):   {_arr_str(qpos_after_servo)}")
        print(f"after_servo_aa(mm/deg):  {_arr_str(pose_after_servo_deg)}")
        print(f"servo_fk_pose_aa(mm/deg): {_arr_str(servo_fk_pose_deg)}")
        print(f"IK_FK(URDF) error : pos_err_mm={ik_fk_err_mm:.3f}, rot_err_rad={ik_fk_err_rad:.6f}")
        print(f"IK+set_angle_j     : pos_err_mm={err_ik_mm:.3f}, rot_err_rad={err_ik_rad:.6f}")
        print(f"set_servo_cartesian: pos_err_mm={err_cart_mm:.3f}, rot_err_rad={err_cart_rad:.6f}")
        print(f"SERVO_FK(URDF) err : pos_err_mm={servo_fk_err_mm:.3f}, rot_err_rad={servo_fk_err_rad:.6f}")

        return {
            "initial_qpos": init_qpos.copy(),
            "initial_pose_aa": init_pose_aa.copy(),
            "input_pose_aa": np.asarray(pose_aa, dtype=np.float64).copy(),
            "ik_action_qpos": ik_action_qpos.copy(),
            "ik_fk_pose_aa": ik_fk_pose_aa.copy(),
            "after_angle_qpos": qpos_after_angle.copy(),
            "after_angle_pose_aa": pose_after_angle_aa.copy(),
            "after_servo_qpos": qpos_after_servo.copy(),
            "after_servo_pose_aa": pose_after_servo_aa.copy(),
            "servo_fk_pose_aa": servo_fk_pose_aa.copy(),
            "after_return_pose_aa": pose_back_aa.copy(),
            "ik_fk_pos_err_mm": float(ik_fk_err_mm),
            "ik_fk_rot_err_rad": float(ik_fk_err_rad),
            "servo_fk_pos_err_mm": float(servo_fk_err_mm),
            "servo_fk_rot_err_rad": float(servo_fk_err_rad),
            "ik_pos_err_mm": float(err_ik_mm),
            "ik_rot_err_rad": float(err_ik_rad),
            "servo_pos_err_mm": float(err_cart_mm),
            "servo_rot_err_rad": float(err_cart_rad),
        }

    finally:
        if arm_ik is not None:
            arm_ik.end()
        if arm_angle is not None:
            arm_angle.end()
        if arm_cart is not None:
            arm_cart.end()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default="ik_angle_j",
        choices=["ik_angle_j", "cartesian_aa", "compare"],
        help="Command method: IK+set_servo_angle_j, direct set_servo_cartesian_aa, or compare",
    )
    parser.add_argument(
        "--pose",
        type=str,
        default=None,
        help='One-shot 6D pose string, e.g. "300 0 250 0 0 0"',
    )
    parser.add_argument("--hold-sec", type=float, default=0.2, help="Wait time after move before readback")
    parser.add_argument("--ee-frame", type=str, default="link6", help="Pinocchio IK end-effector frame")
    parser.add_argument("--fps", type=float, default=100.0, help="Controller loop FPS")
    parser.add_argument("--print-precision", type=int, default=6, help="Decimal digits for console output")
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Output .npz path for compare results. Default: ./xarm_compare_<timestamp>.npz",
    )

    args = parser.parse_args()
    global PRINT_PRECISION
    PRINT_PRECISION = max(0, int(args.print_precision))

    try:
        if args.method == "compare":
            if args.pose is None:
                raise ValueError("--method compare requires --pose")
            pose = parse_pose_tokens(args.pose.replace(",", " ").split())
            pose_deg = pose.copy()
            pose_deg[3:6] = np.rad2deg(pose_deg[3:6])
            print(f"Target pose (mm/deg): {_arr_str(pose_deg)}")
            result = run_sequential_compare(
                pose_aa=pose,
                hold_sec=args.hold_sec,
                ee_frame=args.ee_frame,
                fps=args.fps,
            )
            if args.save_path is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"xarm_compare_{ts}.npz"
            else:
                save_path = args.save_path
            np.savez(
                save_path,
                initial_qpos=result["initial_qpos"],
                initial_pose_aa=result["initial_pose_aa"],
                input_pose_aa=result["input_pose_aa"],
                ik_action_qpos=result["ik_action_qpos"],
                ik_fk_pose_aa=result["ik_fk_pose_aa"],
                after_angle_qpos=result["after_angle_qpos"],
                after_angle_pose_aa=result["after_angle_pose_aa"],
                after_servo_qpos=result["after_servo_qpos"],
                after_servo_pose_aa=result["after_servo_pose_aa"],
                servo_fk_pose_aa=result["servo_fk_pose_aa"],
                after_return_pose_aa=result["after_return_pose_aa"],
                ik_fk_pos_err_mm=result["ik_fk_pos_err_mm"],
                ik_fk_rot_err_rad=result["ik_fk_rot_err_rad"],
                servo_fk_pos_err_mm=result["servo_fk_pos_err_mm"],
                servo_fk_rot_err_rad=result["servo_fk_rot_err_rad"],
                ik_pos_err_mm=result["ik_pos_err_mm"],
                ik_rot_err_rad=result["ik_rot_err_rad"],
                servo_pos_err_mm=result["servo_pos_err_mm"],
                servo_rot_err_rad=result["servo_rot_err_rad"],
            )
            print(f"[SAVE] {save_path}")
            return 0

        arm = build_arm(args.method, ee_frame=args.ee_frame, fps=args.fps)
        time.sleep(0.2)

        if args.pose is not None:
            pose = parse_pose_tokens(args.pose.replace(",", " ").split())
            run_once(arm, pose, args.hold_sec)
        else:
            interactive_loop(arm, args.hold_sec)

    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        if "arm" in locals() and arm is not None:
            arm.end()

    return 0


if __name__ == "__main__":
    sys.exit(main())
