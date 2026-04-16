from pathlib import Path
import sys
import os
import json
import argparse
from typing import Dict, Any, List

import numpy as np
import torch
import trimesh

PROJECT_ROOT = Path(__file__).parents[2]
CUROBO_SRC = PROJECT_ROOT / "curobo" / "src"
if str(CUROBO_SRC) not in sys.path:
    sys.path.insert(0, str(CUROBO_SRC))

from paradex.utils.file_io import load_yaml
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.wrap.reacher.types import ReacherSolveType
from paradex.visualization.robot import RobotModule
from paradex.visualization.visualizer.viser import ViserViewer

DEFAULT_INIT_ARM_QPOS = np.array(
    [-1.481784, -0.445059, -0.71733, 2.073451, -0.033161, 2.759586],
    dtype=np.float32,
)


def _rotmat_to_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    n = np.linalg.norm(q)
    if n > 1e-8:
        q /= n
    return q


def _parse_vec3(raw: str, flag_name: str) -> np.ndarray:
    vals = [x.strip() for x in str(raw).split(",")]
    if len(vals) != 3:
        raise ValueError(f"{flag_name} must be 'x,y,z'")
    return np.array([float(vals[0]), float(vals[1]), float(vals[2])], dtype=np.float32)


def _rot_x(rad: float) -> np.ndarray:
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def _rot_y(rad: float) -> np.ndarray:
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _rot_z(rad: float) -> np.ndarray:
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _compose_delta_rpy_xyz(delta_xyz: np.ndarray, delta_rpy_deg: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    rx, ry, rz = np.deg2rad(delta_rpy_deg.astype(np.float32))
    out[:3, :3] = _rot_z(float(rz)) @ _rot_y(float(ry)) @ _rot_x(float(rx))
    out[:3, 3] = np.asarray(delta_xyz, dtype=np.float32).reshape(3)
    return out


def _normalize_robot_cfg_paths(robot_cfg: Dict[str, Any], cfg_file: str) -> Dict[str, Any]:
    kin = robot_cfg["robot_cfg"]["kinematics"]
    urdf_path = str(kin.get("urdf_path", ""))
    ext_asset = str(kin.get("external_asset_path", "") or "")
    asset_root = str(kin.get("asset_root_path", "") or "")
    cfg_dir = os.path.abspath(os.path.dirname(cfg_file))
    candidates = []
    if os.path.isabs(urdf_path):
        candidates.append(urdf_path)
    if ext_asset:
        candidates.append(os.path.abspath(os.path.join(os.path.expanduser(ext_asset), urdf_path)))
    if asset_root:
        candidates.append(os.path.abspath(os.path.join(os.path.expanduser(asset_root), urdf_path)))
    candidates.append(os.path.abspath(os.path.join(cfg_dir, urdf_path)))

    resolved_urdf = None
    for p in candidates:
        if os.path.exists(p):
            resolved_urdf = p
            break
    if resolved_urdf is None:
        raise FileNotFoundError(f"Cannot resolve URDF path '{urdf_path}' from robot cfg '{cfg_file}'")
    kin["urdf_path"] = resolved_urdf

    # Also resolve collision_spheres path to absolute path.
    coll_spheres = kin.get("collision_spheres")
    if isinstance(coll_spheres, str) and len(coll_spheres) > 0:
        if os.path.isabs(coll_spheres) and os.path.exists(coll_spheres):
            return robot_cfg
        coll_candidates = [
            os.path.abspath(os.path.join(cfg_dir, coll_spheres)),
            os.path.abspath(coll_spheres),
        ]
        for b in (ext_asset, asset_root):
            if b:
                coll_candidates.append(os.path.abspath(os.path.join(os.path.expanduser(b), coll_spheres)))
        for p in coll_candidates:
            if os.path.exists(p):
                kin["collision_spheres"] = p
                break

    return robot_cfg


def _run_ik(
    robot_cfg: Dict[str, Any],
    world_dict: Dict[str, Any],
    start_qpos: np.ndarray,
    target_pose_4x4: np.ndarray,
    seeds: int,
    self_collision: bool,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor_args = TensorDeviceType(device=device)
    mg_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        WorldConfig.from_dict(world_dict),
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        use_cuda_graph=False,
        interpolation_dt=0.01,
        num_ik_seeds=int(seeds),
        num_trajopt_seeds=256,
        num_graph_seeds=32,
        position_threshold=0.01,
        rotation_threshold=0.1,
        self_collision_check=self_collision,
        self_collision_opt=self_collision,
        store_ik_debug=True,
        store_debug_in_result=True,
    )
    motion_gen = MotionGen(mg_cfg)
    motion_gen.warmup(enable_graph=False, warmup_js_trajopt=False)

    active_joint_names = list(getattr(motion_gen.kinematics, "joint_names", []))
    active_dof = len(active_joint_names) if active_joint_names else int(np.asarray(start_qpos).reshape(-1).shape[0])
    sq = np.asarray(start_qpos, dtype=np.float32).reshape(-1)
    if sq.shape[0] != active_dof:
        if sq.shape[0] > active_dof:
            sq = sq[:active_dof]
        else:
            raise ValueError(f"start_qpos dof mismatch: start={sq.shape[0]}, active={active_dof}")

    start = JointState.from_position(
        torch.tensor(sq, device=tensor_args.device, dtype=torch.float32),
        joint_names=active_joint_names if len(active_joint_names) > 0 else None,
    ).unsqueeze(0)

    goal = Pose(
        position=torch.tensor(target_pose_4x4[:3, 3], device=tensor_args.device, dtype=torch.float32).view(1, 3),
        quaternion=torch.tensor(
            _rotmat_to_wxyz(target_pose_4x4[:3, :3]),
            device=tensor_args.device,
            dtype=torch.float32,
        ).view(1, 4),
    )
    plan_cfg = MotionGenPlanConfig(
        enable_graph=False,
        max_attempts=1,
        enable_graph_attempt=None,
        enable_finetune_trajopt=False,
        num_ik_seeds=int(seeds),
        num_trajopt_seeds=64,
    )
    solve_state = motion_gen._get_solve_state(ReacherSolveType.SINGLE, plan_cfg, goal, start)
    result = motion_gen._solve_ik_from_solve_state(
        goal_pose=goal,
        solve_state=solve_state,
        start_state=start,
        use_nn_seed=False,
        partial_ik_opt=False,
        link_poses=None,
    )

    succ = result.success.detach().cpu().reshape(-1).numpy().astype(bool)
    pe = result.position_error.detach().cpu().reshape(-1).numpy().astype(np.float64)
    re = result.rotation_error.detach().cpu().reshape(-1).numpy().astype(np.float64)
    score = pe + re
    if np.any(succ):
        valid_idx = np.where(succ)[0]
        best_idx = int(valid_idx[np.argmin(score[valid_idx])])
        success = True
    else:
        best_idx = int(np.argmin(score))
        success = False
    pos_err = float(pe[best_idx])
    rot_err = float(re[best_idx])
    q_sol = None
    if success:
        sol = result.solution.detach().cpu().numpy().astype(np.float32).reshape(-1, active_dof)
        q_sol = sol[best_idx]
    return success, pos_err, rot_err, q_sol


def _load_robot_cfg_for_curobo(robot_cfg_file: str) -> Dict[str, Any]:
    cfg = load_yaml(robot_cfg_file)
    return _normalize_robot_cfg_paths(cfg, robot_cfg_file)


def _make_arm_only_robot_cfg(
    robot_cfg: Dict[str, Any], start_qpos: np.ndarray, joint_names: List[str], arm_dof: int = 6
) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(robot_cfg))  # simple deep copy
    kin = cfg["robot_cfg"]["kinematics"]
    cspace = kin.setdefault("cspace", {})
    arm_names = list(joint_names[:arm_dof])
    hand_names = list(joint_names[arm_dof:])
    cspace["joint_names"] = arm_names

    for k in ("retract_config", "null_space_weight", "cspace_distance_weight"):
        if isinstance(cspace.get(k), list) and len(cspace[k]) >= arm_dof:
            cspace[k] = cspace[k][:arm_dof]
        elif k == "retract_config":
            cspace[k] = [0.0] * arm_dof
        else:
            cspace[k] = [1.0] * arm_dof

    lock = kin.get("lock_joints") or {}
    for i, jn in enumerate(hand_names):
        idx = arm_dof + i
        if idx < len(start_qpos):
            lock[jn] = float(start_qpos[idx])
    kin["lock_joints"] = lock
    return cfg


def _expand_to_full_qpos(active_q: np.ndarray, start_q: np.ndarray) -> np.ndarray:
    out = np.asarray(start_q, dtype=np.float32).copy()
    aq = np.asarray(active_q, dtype=np.float32).reshape(-1)
    n = min(aq.shape[0], out.shape[0])
    out[:n] = aq[:n]
    return out


def _run_motiongen(
    robot_cfg: Dict[str, Any],
    world_dict: Dict[str, Any],
    start_qpos: np.ndarray,
    target_pose_4x4: np.ndarray,
    seeds: int,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor_args = TensorDeviceType(device=device)

    mg_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        WorldConfig.from_dict(world_dict),
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        use_cuda_graph=False,
        interpolation_dt=0.01,
        num_ik_seeds=int(seeds),
        num_trajopt_seeds=64,
        num_graph_seeds=32,
        position_threshold=0.01,
        rotation_threshold=0.1,
        store_ik_debug=True,
        store_debug_in_result=True,
    )
    motion_gen = MotionGen(mg_cfg)
    motion_gen.warmup(enable_graph=False, warmup_js_trajopt=False)

    active_joint_names = list(getattr(motion_gen.kinematics, "joint_names", []))
    active_dof = len(active_joint_names) if active_joint_names else int(np.asarray(start_qpos).reshape(-1).shape[0])
    sq = np.asarray(start_qpos, dtype=np.float32).reshape(-1)
    if sq.shape[0] != active_dof:
        if sq.shape[0] > active_dof:
            sq = sq[:active_dof]
        else:
            raise ValueError(f"start_qpos dof mismatch: start={sq.shape[0]}, active={active_dof}")

    start = JointState.from_position(
        torch.tensor(sq, device=tensor_args.device, dtype=torch.float32),
        joint_names=active_joint_names if len(active_joint_names) > 0 else None,
    ).unsqueeze(0)
    goal = Pose(
        position=torch.tensor(target_pose_4x4[:3, 3], device=tensor_args.device, dtype=torch.float32).view(1, 3),
        quaternion=torch.tensor(_rotmat_to_wxyz(target_pose_4x4[:3, :3]), device=tensor_args.device, dtype=torch.float32).view(1, 4),
    )
    plan_cfg = MotionGenPlanConfig(enable_graph=False, max_attempts=20, enable_graph_attempt=3, enable_finetune_trajopt=True)
    result = motion_gen.plan_single(start, goal, plan_cfg)
    success = bool(result.success.item()) if hasattr(result.success, "item") else bool(result.success)

    out = {
        "success": bool(success),
        "status": str(result.status),
        "active_dof": int(active_dof),
        "active_joint_names": active_joint_names,
        "fallback_used": False,
        "trajectory_active": None,
    }

    if not success and str(result.status) == "MotionGenStatus.IK_FAIL":
        ik_result = motion_gen.ik_solver.solve_single(goal)
        ik_success = bool(ik_result.success.item()) if hasattr(ik_result.success, "item") else bool(ik_result.success)
        out["ik_solver_success"] = bool(ik_success)
        if ik_success:
            goal_q = ik_result.solution[ik_result.success].view(-1, active_dof)[0]
            goal_js = JointState.from_position(
                goal_q,
                joint_names=active_joint_names if len(active_joint_names) > 0 else None,
            ).unsqueeze(0)
            js_result = motion_gen.plan_single_js(start, goal_js, plan_cfg)
            js_success = bool(js_result.success.item()) if hasattr(js_result.success, "item") else bool(js_result.success)
            out["fallback_status"] = str(js_result.status)
            out["fallback_used"] = True
            if js_success:
                result = js_result
                success = True
                out["success"] = True
                out["status"] = str(js_result.status)

    if success:
        interp = result.get_interpolated_plan()
        out["trajectory_active"] = interp.position.detach().cpu().numpy().astype(np.float32)

    return out


def _load_mesh(mesh_path: str) -> trimesh.Trimesh:
    geom = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(geom, trimesh.Trimesh):
        return geom
    if isinstance(geom, trimesh.Scene):
        geoms = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"no mesh geometry in scene: {mesh_path}")
        return trimesh.util.concatenate(geoms)
    if isinstance(geom, list):
        return trimesh.util.concatenate(geom)
    raise ValueError(f"unsupported mesh type: {type(geom)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--planning_scene",
        required=True,
        help="path to planning_scene_curobo.json",
    )
    parser.add_argument(
        "--robot_cfg",
        default=str(PROJECT_ROOT / "rsc" / "curobo" / "content" / "configs" / "robot" / "xarm_inspire.yml"),
    )
    parser.add_argument("--ee_link", default="link6")
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--arm_only", action="store_true", help="use arm-only cspace (6 dof)")
    parser.add_argument("--motiongen", action="store_true", help="run MotionGen planning")
    parser.add_argument("--ignore_world", action="store_true", help="disable world collision in MotionGen")
    parser.add_argument(
        "--motiongen_use_start_pose_target",
        action="store_true",
        help="debug: replace MotionGen target with (start ee pose + small delta)",
    )
    parser.add_argument(
        "--motiongen_target_delta_xyz",
        default="0.0,0.0,0.01",
        help="debug target translation delta in meters: dx,dy,dz",
    )
    parser.add_argument(
        "--motiongen_target_delta_rpy_deg",
        default="0.0,0.0,0.0",
        help="debug target rotation delta in degrees: droll,dpitch,dyaw",
    )
    parser.add_argument("--vis", action="store_true", help="open viser and show start/ik robot + target frame")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cuRobo IK solver in this setup.")

    with open(args.planning_scene, "r", encoding="utf-8") as f:
        scene = json.load(f)
    target_pose = np.asarray(scene["target"]["target_eef_pose_robot"], dtype=np.float32)
    vis_target_pose = target_pose.copy()
    start_qpos = np.asarray(scene["target"].get("start_qpos", []), dtype=np.float32).reshape(-1)
    if start_qpos.size == 0:
        start_qpos = np.zeros(12, dtype=np.float32)
        n_arm = min(6, start_qpos.shape[0])
        start_qpos[:n_arm] = DEFAULT_INIT_ARM_QPOS[:n_arm]
    robot_cfg = _load_robot_cfg_for_curobo(os.path.expanduser(args.robot_cfg))
    if args.arm_only:
        robot_module = RobotModule(robot_cfg["robot_cfg"]["kinematics"]["urdf_path"])
        robot_cfg = _make_arm_only_robot_cfg(
            robot_cfg=robot_cfg,
            start_qpos=start_qpos,
            joint_names=list(robot_module.get_joint_names()),
            arm_dof=6,
        )
    world_dict = {"cuboid": {}, "mesh": {}} if args.ignore_world else scene.get("world", {"cuboid": {}, "mesh": {}})

    print(f"planning_scene: {args.planning_scene}")
    print(f"robot_cfg: {args.robot_cfg}")
    print(f"ee_link: {args.ee_link}")
    print(f"target position: {target_pose[:3, 3].tolist()}")
    print(f"target quaternion(wxyz): {_rotmat_to_wxyz(target_pose[:3, :3]).tolist()}")

    best_solution = None
    motiongen_out = None
    for self_collision in (False, True):
        success, pos_err, rot_err, q_sol = _run_ik(
            robot_cfg=robot_cfg,
            world_dict=world_dict,
            start_qpos=start_qpos,
            target_pose_4x4=target_pose,
            seeds=int(args.seeds),
            self_collision=self_collision,
        )
        print(
            f"self_collision={self_collision} | success={success} | "
            f"position_error={pos_err:.6f} | rotation_error={rot_err:.6f}"
        )
        if q_sol is not None:
            print("q_solution:", q_sol.tolist())
            if best_solution is None:
                best_solution = q_sol

    if args.motiongen:
        target_pose_motiongen = target_pose.copy()
        if args.motiongen_use_start_pose_target:
            robot_urdf_path = scene.get("robot", {}).get("urdf_path")
            if robot_urdf_path is None or not os.path.exists(robot_urdf_path):
                raise FileNotFoundError(f"robot urdf_path missing/invalid in planning scene: {robot_urdf_path}")
            robot_module = RobotModule(robot_urdf_path)
            if start_qpos.shape[0] != robot_module.get_num_joints():
                raise ValueError(
                    f"start_qpos length mismatch for FK: start={start_qpos.shape[0]}, robot={robot_module.get_num_joints()}"
                )
            robot_module.update_cfg(start_qpos)
            start_eef_pose = np.asarray(
                robot_module.get_transform(args.ee_link, robot_module.urdf.base_link, collision_geometry=False),
                dtype=np.float32,
            )
            delta_xyz = _parse_vec3(args.motiongen_target_delta_xyz, "--motiongen_target_delta_xyz")
            delta_rpy_deg = _parse_vec3(args.motiongen_target_delta_rpy_deg, "--motiongen_target_delta_rpy_deg")
            delta_tf = _compose_delta_rpy_xyz(delta_xyz, delta_rpy_deg)
            target_pose_motiongen = start_eef_pose @ delta_tf
            print("motiongen debug target enabled:")
            print("  start ee position:", start_eef_pose[:3, 3].tolist())
            print("  delta xyz [m]:", delta_xyz.tolist())
            print("  delta rpy [deg]:", delta_rpy_deg.tolist())
            print("  debug target position:", target_pose_motiongen[:3, 3].tolist())
            print("  debug target quaternion(wxyz):", _rotmat_to_wxyz(target_pose_motiongen[:3, :3]).tolist())
        vis_target_pose = target_pose_motiongen

        motiongen_out = _run_motiongen(
            robot_cfg=robot_cfg,
            world_dict=world_dict,
            start_qpos=start_qpos,
            target_pose_4x4=target_pose_motiongen,
            seeds=int(args.seeds),
        )
        print(
            f"motiongen | success={motiongen_out['success']} | status={motiongen_out['status']} | "
            f"active_dof={motiongen_out['active_dof']} | fallback_used={motiongen_out['fallback_used']}"
        )
        if "ik_solver_success" in motiongen_out:
            print(f"motiongen fallback ik_solver_success={motiongen_out['ik_solver_success']}")
        if motiongen_out.get("fallback_status") is not None:
            print(f"motiongen fallback_status={motiongen_out['fallback_status']}")

    if args.vis:
        robot_urdf_path = scene.get("robot", {}).get("urdf_path")
        if robot_urdf_path is None or not os.path.exists(robot_urdf_path):
            raise FileNotFoundError(f"robot urdf_path missing/invalid in planning scene: {robot_urdf_path}")

        vis = ViserViewer()
        vis.add_floor(height=0.0)
        vis.add_robot("robot_start", robot_urdf_path)
        if start_qpos.size > 0:
            vis.robot_dict["robot_start"].update_cfg(start_qpos)

        if motiongen_out is not None and motiongen_out.get("trajectory_active") is not None:
            traj_active = np.asarray(motiongen_out["trajectory_active"], dtype=np.float32)
            if start_qpos.size > 0 and traj_active.shape[-1] != start_qpos.shape[0]:
                traj_full = np.stack([_expand_to_full_qpos(q, start_qpos) for q in traj_active], axis=0)
            else:
                traj_full = traj_active
            vis.add_traj("motiongen_traj", {"robot_start": traj_full}, {})

        if best_solution is not None:
            vis.add_robot("robot_ik_solution", robot_urdf_path)
            if start_qpos.size > 0 and best_solution.shape[0] == 6 and start_qpos.shape[0] > 6:
                q_full = start_qpos.copy()
                q_full[:6] = best_solution[:6]
                vis.robot_dict["robot_ik_solution"].update_cfg(q_full)
            else:
                vis.robot_dict["robot_ik_solution"].update_cfg(best_solution)

        # target frame (use the same target actually used for planning/debug)
        target_q = _rotmat_to_wxyz(vis_target_pose[:3, :3])
        vis.server.scene.add_frame(
            "/target/eef_pose",
            position=vis_target_pose[:3, 3].astype(float),
            wxyz=target_q.astype(float),
            show_axes=True,
            axes_length=0.08,
            axes_radius=0.004,
        )

        # optional world geometry from planning scene
        world = scene.get("world", {})
        table = world.get("cuboid", {}).get("table")
        if isinstance(table, dict):
            dims = table.get("dims", [1.0, 1.0, 0.1])
            pose = table.get("pose", [0, 0, -0.05, 1, 0, 0, 0])
            vis.server.scene.add_box(
                "/world/table",
                dimensions=tuple(float(x) for x in dims),
                position=(float(pose[0]), float(pose[1]), float(pose[2])),
                color=(160, 130, 100),
            )
        obj = world.get("mesh", {}).get("object")
        if isinstance(obj, dict) and os.path.exists(obj.get("file_path", "")):
            mesh = _load_mesh(obj["file_path"])
            x, y, z, qw, qx, qy, qz = [float(v) for v in obj.get("pose", [0, 0, 0, 1, 0, 0, 0])]
            # wxyz -> rot
            w, a, b, c = qw, qx, qy, qz
            R = np.array(
                [
                    [1 - 2 * (b * b + c * c), 2 * (a * b - c * w), 2 * (a * c + b * w)],
                    [2 * (a * b + c * w), 1 - 2 * (a * a + c * c), 2 * (b * c - a * w)],
                    [2 * (a * c - b * w), 2 * (b * c + a * w), 1 - 2 * (a * a + b * b)],
                ],
                dtype=np.float32,
            )
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3, 3] = np.array([x, y, z], dtype=np.float32)
            vis.add_object("object", mesh, T, opacity=1.0)

        print("Viser running. Press Ctrl+C to exit.")
        try:
            while True:
                vis.update()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
