"""
Run hecalib compare using xArmAPI (no ROS/rclpy dependency).

For each system/current/hecalib/xarm/{i}_aa.npy pose:
1) save initial qpos/pose
2) IK -> set_servo_angle_j
3) read after-angle pose/qpos and IK-FK error
4) return to initial qpos by set_servo_angle_j
5) set_servo_cartesian_aa to target
6) read after-servo pose/qpos and SERVO-FK error
7) return to initial pose by set_servo_cartesian_aa

Outputs one npz per index + summary.
"""

import argparse
import glob
import os
import time
from datetime import datetime

import numpy as np
import transforms3d as t3d
from scipy.spatial.transform import Rotation
import pinocchio as pin
from xarm.wrapper import XArmAPI


def aa2homo(aa):
    h = np.eye(4, dtype=np.float64)
    h[:3, 3] = np.asarray(aa[:3], dtype=np.float64) / 1000.0
    rv = np.asarray(aa[3:6], dtype=np.float64)
    ang = np.linalg.norm(rv)
    if ang < 1e-12:
        h[:3, :3] = np.eye(3, dtype=np.float64)
    else:
        h[:3, :3] = t3d.axangles.axangle2mat(rv / ang, ang)
    return h


def homo2aa(h):
    t = h[:3, 3] * 1000.0
    ax, ang = t3d.axangles.mat2axangle(h[:3, :3], unit_thresh=0.001)
    return np.concatenate([t, np.asarray(ax, dtype=np.float64) * float(ang)])


def cart2homo(cart):
    h = np.eye(4, dtype=np.float64)
    h[:3, 3] = np.asarray(cart[:3], dtype=np.float64) / 1000.0
    h[:3, :3] = Rotation.from_euler("xyz", np.asarray(cart[3:6], dtype=np.float64)).as_matrix()
    return h


def pose_error(target_h, current_h):
    pos_mm = np.linalg.norm((current_h[:3, 3] - target_h[:3, 3]) * 1000.0)
    dR = target_h[:3, :3].T @ current_h[:3, :3]
    tr = np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0)
    rot_rad = np.arccos(tr)
    return float(pos_mm), float(rot_rad)


class PinocchioIKSolver:
    def __init__(self, urdf_path, ee_frame="link_tcp", max_iter=200, tol=1e-4, step_size=0.5, damping=1e-5):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(ee_frame)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.step_size = float(step_size)
        self.damping = float(damping)
        self.q_min = np.asarray(self.model.lowerPositionLimit, dtype=np.float64)
        self.q_max = np.asarray(self.model.upperPositionLimit, dtype=np.float64)

    def solve(self, target_homo, q_init):
        q = np.asarray(q_init, dtype=np.float64).copy()
        q = np.clip(q, self.q_min, self.q_max)
        target = pin.SE3(target_homo[:3, :3], target_homo[:3, 3])

        err_norm = np.inf
        for i in range(self.max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            current = pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
            err = pin.log(current.inverse() * target).vector
            err_norm = float(np.linalg.norm(err))
            if err_norm < self.tol:
                return q, True, i + 1, err_norm

            J = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL
            )
            curr_damping = self.damping if err_norm > 0.01 else self.damping * 0.1
            lhs = J.T @ J + curr_damping * np.eye(self.model.nv, dtype=np.float64)
            rhs = J.T @ err
            dq = np.linalg.solve(lhs, rhs)

            q = pin.integrate(self.model, q, dq * self.step_size)
            q = np.clip(q, self.q_min, self.q_max)

        return q, False, self.max_iter, err_norm

    def fk(self, q):
        q = np.asarray(q, dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, q)
        pose = pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        return pose.homogeneous


class XArmCompareSDK:
    def __init__(self, ip, urdf_path, ee_frame="link_tcp"):
        self.arm = XArmAPI(ip, report_type="devlop")
        if self.arm.has_err_warn:
            self.arm.clean_warn()
            self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)
        self.arm.set_state(state=0)
        time.sleep(0.1)

        self.ik = PinocchioIKSolver(urdf_path=urdf_path, ee_frame=ee_frame)

    def get_qpos(self):
        code, state = self.arm.get_joint_states(is_radian=True)
        if code != 0:
            raise RuntimeError(f"get_joint_states failed: {code}")
        return np.asarray(state[0], dtype=np.float64)[:6]

    def get_pose_homo(self):
        code, cart = self.arm.get_position(is_radian=True)
        if code != 0:
            raise RuntimeError(f"get_position failed: {code}")
        return cart2homo(np.asarray(cart, dtype=np.float64)[:6])

    def set_servo_angle_j(self, q):
        ret = self.arm.set_servo_angle_j(angles=np.asarray(q, dtype=np.float64).tolist(), is_radian=True)
        if ret != 0:
            raise RuntimeError(f"set_servo_angle_j failed: {ret}")

    def set_servo_cartesian_aa(self, aa):
        ret = self.arm.set_servo_cartesian_aa(np.asarray(aa, dtype=np.float64).tolist(), is_radian=True)
        if ret != 0:
            raise RuntimeError(f"set_servo_cartesian_aa failed: {ret}")

    def compare_once(self, target_aa, hold_sec=1.0):
        target_h = aa2homo(target_aa)

        initial_qpos = self.get_qpos()
        initial_pose_h = self.get_pose_homo()
        initial_pose_aa = homo2aa(initial_pose_h)

        ik_q, succ, _, _ = self.ik.solve(target_h, initial_qpos)
        if not succ:
            raise RuntimeError("IK failed")

        self.set_servo_angle_j(ik_q)
        time.sleep(hold_sec)
        after_angle_qpos = self.get_qpos()
        after_angle_h = self.get_pose_homo()
        after_angle_aa = homo2aa(after_angle_h)
        ik_fk_h = self.ik.fk(ik_q)
        ik_fk_aa = homo2aa(ik_fk_h)
        ik_fk_pos_err_mm, ik_fk_rot_err_rad = pose_error(target_h, ik_fk_h)
        ik_pos_err_mm, ik_rot_err_rad = pose_error(target_h, after_angle_h)

        self.set_servo_angle_j(initial_qpos)
        time.sleep(hold_sec)

        self.set_servo_cartesian_aa(target_aa)
        time.sleep(hold_sec)
        after_servo_qpos = self.get_qpos()
        after_servo_h = self.get_pose_homo()
        after_servo_aa = homo2aa(after_servo_h)
        servo_fk_h = self.ik.fk(after_servo_qpos)
        servo_fk_aa = homo2aa(servo_fk_h)
        servo_fk_pos_err_mm, servo_fk_rot_err_rad = pose_error(target_h, servo_fk_h)
        servo_pos_err_mm, servo_rot_err_rad = pose_error(target_h, after_servo_h)

        self.set_servo_cartesian_aa(initial_pose_aa)
        time.sleep(hold_sec)
        after_return_aa = homo2aa(self.get_pose_homo())

        return {
            "initial_qpos": initial_qpos,
            "initial_pose_aa": initial_pose_aa,
            "input_pose_aa": np.asarray(target_aa, dtype=np.float64),
            "ik_action_qpos": ik_q,
            "ik_fk_pose_aa": ik_fk_aa,
            "after_angle_qpos": after_angle_qpos,
            "after_angle_pose_aa": after_angle_aa,
            "after_servo_qpos": after_servo_qpos,
            "after_servo_pose_aa": after_servo_aa,
            "servo_fk_pose_aa": servo_fk_aa,
            "after_return_pose_aa": after_return_aa,
            "ik_fk_pos_err_mm": float(ik_fk_pos_err_mm),
            "ik_fk_rot_err_rad": float(ik_fk_rot_err_rad),
            "ik_pos_err_mm": float(ik_pos_err_mm),
            "ik_rot_err_rad": float(ik_rot_err_rad),
            "servo_fk_pos_err_mm": float(servo_fk_pos_err_mm),
            "servo_fk_rot_err_rad": float(servo_fk_rot_err_rad),
            "servo_pos_err_mm": float(servo_pos_err_mm),
            "servo_rot_err_rad": float(servo_rot_err_rad),
        }

    def end(self):
        if self.arm is not None:
            self.arm.disconnect()


def collect_pose_files(root, start_idx, end_idx):
    files = []
    for i in range(start_idx, end_idx + 1):
        p = os.path.join(root, f"{i}_aa.npy")
        if os.path.exists(p):
            files.append((i, p))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.1.221")
    parser.add_argument("--urdf", type=str, default="rsc/robot/xarm/xarm.urdf")
    parser.add_argument("--ee-frame", type=str, default="link_tcp")
    parser.add_argument("--pose-root", type=str, default="system/current/hecalib/xarm")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--hold-sec", type=float, default=1.5)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    files = collect_pose_files(args.pose_root, args.start, args.end)
    print(f"found {len(files)} pose files")
    if not files:
        raise RuntimeError("No pose files found")

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"xarm_hecalib_compare_sdk_{ts}"
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print("output dir:", out_dir)

    runner = XArmCompareSDK(ip=args.ip, urdf_path=args.urdf, ee_frame=args.ee_frame)

    successes = 0
    failures = []
    pos_errs = []
    rot_errs = []

    try:
        for n, (idx, path) in enumerate(files, start=1):
            print(f"\n[{n}/{len(files)}] idx={idx} file={path}")
            try:
                h = np.asarray(np.load(path), dtype=np.float64)
                if h.shape != (4, 4):
                    raise ValueError(f"Unexpected pose shape: {h.shape}")
                pose_aa = homo2aa(h)
                pose_deg = pose_aa.copy()
                pose_deg[3:6] = np.rad2deg(pose_deg[3:6])
                print(f"target(mm/deg): {np.array2string(pose_deg, precision=3)}")

                result = runner.compare_once(pose_aa, hold_sec=args.hold_sec)

                out_path = os.path.join(out_dir, f"{idx:02d}_compare.npz")
                np.savez(out_path, source_pose_file=path, source_index=np.array([idx], dtype=np.int32), **result)
                print(f"saved: {out_path}")

                successes += 1
                pos_errs.append(float(result["servo_fk_pos_err_mm"]))
                rot_errs.append(float(result["servo_fk_rot_err_rad"]))
            except Exception as e:
                failures.append((idx, path, str(e)))
                print(f"[FAIL] idx={idx}: {e}")
    finally:
        runner.end()

    print("\n=== Batch Summary ===")
    print(f"success={successes}, fail={len(failures)}")
    if pos_errs:
        x = np.asarray(pos_errs, dtype=np.float64)
        print(f"servo_fk_pos_err_mm: mean={np.mean(x):.4f}, median={np.median(x):.4f}, p95={np.percentile(x,95):.4f}, max={np.max(x):.4f}")
    if rot_errs:
        x = np.asarray(rot_errs, dtype=np.float64)
        print(f"servo_fk_rot_err_rad: mean={np.mean(x):.6f}, median={np.median(x):.6f}, p95={np.percentile(x,95):.6f}, max={np.max(x):.6f}")

    if failures:
        fail_path = os.path.join(out_dir, "failures.txt")
        with open(fail_path, "w", encoding="utf-8") as f:
            for idx, path, msg in failures:
                f.write(f"{idx}\t{path}\t{msg}\n")
        print(f"failure log: {fail_path}")


if __name__ == "__main__":
    main()
