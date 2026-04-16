from pathlib import Path
import sys
import os
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple

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


def _parse_qpos(raw: str, flag_name: str) -> np.ndarray:
    vals = [x.strip() for x in str(raw).split(",")]
    if not vals:
        raise ValueError(f"{flag_name} must be comma-separated floats")
    return np.array([float(v) for v in vals], dtype=np.float32)


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


def _normalize_robot_cfg_paths(robot_cfg: Dict[str, Any], cfg_file: str) -> Dict[str, Any]:
    kin = robot_cfg["robot_cfg"]["kinematics"]
    urdf_path = str(kin.get("urdf_path", ""))
    ext_asset = str(kin.get("external_asset_path", "") or "")
    asset_root = str(kin.get("asset_root_path", "") or "")

    if os.path.isabs(urdf_path) and os.path.exists(urdf_path):
        return robot_cfg

    cfg_dir = os.path.abspath(os.path.dirname(cfg_file))
    base_dirs = [cfg_dir]
    p = Path(cfg_dir)
    for i in range(1, min(8, len(p.parents) + 1)):
        base_dirs.append(str(p.parents[i - 1]))
    base_dirs.append(str(PROJECT_ROOT))

    urdf_path_exp = os.path.expanduser(urdf_path)
    urdf_base = os.path.basename(urdf_path_exp)
    candidates: List[str] = []

    def _add(path: str) -> None:
        path = os.path.abspath(os.path.expanduser(path))
        if path not in candidates:
            candidates.append(path)

    # Direct interpretations
    _add(os.path.join(cfg_dir, urdf_path_exp))
    _add(urdf_path_exp)

    # Resolve with asset roots relative to multiple possible anchors.
    for b in base_dirs:
        if ext_asset:
            _add(os.path.join(b, ext_asset, urdf_path_exp))
            _add(os.path.join(b, ext_asset, urdf_base))
            _add(os.path.join(b, "assets", ext_asset, urdf_path_exp))
            _add(os.path.join(b, "assets", ext_asset, urdf_base))
        if asset_root:
            _add(os.path.join(b, asset_root, urdf_path_exp))
            _add(os.path.join(b, asset_root, urdf_base))
            _add(os.path.join(b, "assets", asset_root, urdf_path_exp))
            _add(os.path.join(b, "assets", asset_root, urdf_base))
        _add(os.path.join(b, urdf_path_exp))
        _add(os.path.join(b, "assets", urdf_path_exp))

    for p in candidates:
        if os.path.exists(p):
            kin["urdf_path"] = p
            break
    else:
        raise FileNotFoundError(f"Cannot resolve URDF path '{urdf_path}' from robot cfg '{cfg_file}'")

    # Resolve collision_spheres config path as well (often relative like "spheres/xarm_allegro.yml").
    coll_spheres = kin.get("collision_spheres")
    if isinstance(coll_spheres, str) and len(coll_spheres) > 0:
        cs_raw = os.path.expanduser(coll_spheres)
        if not os.path.isabs(cs_raw) or not os.path.exists(cs_raw):
            cs_candidates: List[str] = []

            def _add_cs(path: str) -> None:
                path = os.path.abspath(os.path.expanduser(path))
                if path not in cs_candidates:
                    cs_candidates.append(path)

            cfg_dir = os.path.abspath(os.path.dirname(cfg_file))
            base_dirs = [cfg_dir]
            p = Path(cfg_dir)
            for i in range(1, min(8, len(p.parents) + 1)):
                base_dirs.append(str(p.parents[i - 1]))

            _add_cs(os.path.join(cfg_dir, cs_raw))
            _add_cs(cs_raw)
            for b in base_dirs:
                _add_cs(os.path.join(b, cs_raw))
                _add_cs(os.path.join(b, "configs", "robot", cs_raw))
                _add_cs(os.path.join(b, "content", "configs", "robot", cs_raw))
                _add_cs(os.path.join(b, "assets", "content", "configs", "robot", cs_raw))

            for csp in cs_candidates:
                if os.path.exists(csp):
                    kin["collision_spheres"] = csp
                    break

    return robot_cfg


def _make_arm_only_robot_cfg(
    robot_cfg: Dict[str, Any], start_qpos: np.ndarray, joint_names: List[str], arm_dof: int = 6
) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(robot_cfg))
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


def _resolve_start_qpos(start_qpos_in: np.ndarray, robot_cfg: Dict[str, Any]) -> np.ndarray:
    kin = robot_cfg["robot_cfg"]["kinematics"]
    cspace = kin.get("cspace", {})
    joint_names = list(cspace.get("joint_names", []))
    dof = len(joint_names)
    retract = cspace.get("retract_config")

    sq = np.asarray(start_qpos_in, dtype=np.float32).reshape(-1)
    if dof == 0:
        return sq
    if sq.shape[0] == dof:
        return sq

    out = None
    if isinstance(retract, list) and len(retract) == dof:
        out = np.asarray(retract, dtype=np.float32).copy()
    else:
        out = np.zeros(dof, dtype=np.float32)
    n = min(sq.shape[0], dof)
    out[:n] = sq[:n]
    return out


def _lock_hand_joints(
    robot_cfg: Dict[str, Any],
    qpos_ref: np.ndarray,
    arm_dof: int = 6,
) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(robot_cfg))
    kin = cfg["robot_cfg"]["kinematics"]
    cspace = kin.get("cspace", {})
    joint_names = list(cspace.get("joint_names", []))
    dof = len(joint_names)
    if dof <= arm_dof:
        return cfg

    q = np.asarray(qpos_ref, dtype=np.float32).reshape(-1)
    if q.shape[0] != dof:
        raise ValueError(f"lock_joints qpos dof mismatch: qpos={q.shape[0]}, cspace={dof}")

    lock = kin.get("lock_joints") or {}
    for i, jn in enumerate(joint_names[arm_dof:], start=arm_dof):
        lock[jn] = float(q[i])
    kin["lock_joints"] = lock
    return cfg


def _select_active_qpos_from_cspace(
    qpos_full: np.ndarray, robot_cfg: Dict[str, Any], active_joint_names: List[str]
) -> np.ndarray:
    q_full = np.asarray(qpos_full, dtype=np.float32).reshape(-1)
    if len(active_joint_names) == q_full.shape[0]:
        return q_full

    cspace = robot_cfg.get("robot_cfg", {}).get("kinematics", {}).get("cspace", {})
    cspace_joint_names = list(cspace.get("joint_names", []))
    if len(cspace_joint_names) != q_full.shape[0]:
        raise ValueError(
            f"cannot align qpos to active joints: cspace_names={len(cspace_joint_names)}, qpos={q_full.shape[0]}, active={len(active_joint_names)}"
        )
    qmap = {str(jn): float(q_full[i]) for i, jn in enumerate(cspace_joint_names)}
    out = []
    missing = []
    for jn in active_joint_names:
        if jn not in qmap:
            missing.append(str(jn))
        else:
            out.append(qmap[jn])
    if missing:
        raise ValueError(f"active joint names missing from cspace joint_names: {missing}")
    return np.asarray(out, dtype=np.float32)


def _apply_hand_qpos_config_cost(
    robot_cfg: Dict[str, Any],
    target_qpos: np.ndarray,
    hand_weight: float = 1.0,
    arm_dof: int = 6,
) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(robot_cfg))
    kin = cfg["robot_cfg"]["kinematics"]
    cspace = kin.setdefault("cspace", {})
    joint_names = list(cspace.get("joint_names", []))
    dof = len(joint_names)
    if dof == 0:
        return cfg

    q = np.asarray(target_qpos, dtype=np.float32).reshape(-1)
    if q.shape[0] != dof:
        raise ValueError(f"hand config-cost target dof mismatch: target={q.shape[0]}, cspace={dof}")

    retract = cspace.get("retract_config")
    if not isinstance(retract, list) or len(retract) != dof:
        retract = [0.0] * dof
    for i in range(dof):
        retract[i] = float(q[i])
    cspace["retract_config"] = retract

    ns_weight = [0.0] * dof
    for i in range(arm_dof, dof):
        ns_weight[i] = float(hand_weight)
    cspace["null_space_weight"] = ns_weight

    if not (isinstance(cspace.get("cspace_distance_weight"), list) and len(cspace["cspace_distance_weight"]) == dof):
        cspace["cspace_distance_weight"] = [1.0] * dof
    return cfg


def _sanitize_world_dict(world_dict: Dict[str, Any]) -> Dict[str, Any]:
    world = json.loads(json.dumps(world_dict))
    world.setdefault("cuboid", {})
    world.setdefault("mesh", {})
    if len(world["cuboid"]) == 0 and len(world["mesh"]) == 0:
        world["cuboid"]["__dummy__"] = {
            "dims": [0.01, 0.01, 0.01],
            "pose": [1000.0, 1000.0, 1000.0, 1.0, 0.0, 0.0, 0.0],
        }
    return world


def _qpos_to_cfg_map(joint_names: List[str], q: np.ndarray) -> Dict[str, float]:
    qq = np.asarray(q, dtype=np.float32).reshape(-1)
    return {str(jn): float(qq[i]) for i, jn in enumerate(joint_names[: qq.shape[0]])}


def _build_sphere_link_names(motion_gen: MotionGen) -> List[str]:
    kin_cfg = motion_gen.kinematics.kinematics_config
    link_sphere_idx_map = kin_cfg.link_sphere_idx_map.detach().cpu().numpy().astype(np.int64).reshape(-1)
    idx_to_link = {int(v): str(k) for k, v in (kin_cfg.link_name_to_idx_map or {}).items()}
    return [idx_to_link.get(int(idx), f"link_idx_{int(idx)}") for idx in link_sphere_idx_map]


def _is_ignored_pair(link_a: str, link_b: str, ignore_map: Dict[str, Any]) -> bool:
    if link_a == link_b:
        return True
    a_ign = ignore_map.get(link_a, []) if isinstance(ignore_map, dict) else []
    b_ign = ignore_map.get(link_b, []) if isinstance(ignore_map, dict) else []
    return (link_b in a_ign) or (link_a in b_ign)


def _detect_self_collision_pairs(
    motion_gen: MotionGen,
    state: JointState,
    robot_cfg: Dict[str, Any],
    max_pairs: int = 20,
) -> List[Dict[str, Any]]:
    kin_state = motion_gen.compute_kinematics(state)
    sph_t = None
    if hasattr(kin_state, "robot_spheres") and kin_state.robot_spheres is not None:
        sph_t = kin_state.robot_spheres
    elif hasattr(kin_state, "get_link_spheres"):
        sph_t = kin_state.get_link_spheres()
    elif hasattr(kin_state, "link_spheres_tensor") and kin_state.link_spheres_tensor is not None:
        sph_t = kin_state.link_spheres_tensor
    else:
        raise RuntimeError("Cannot find sphere tensor in kinematics state")

    spheres_np = sph_t.detach().cpu().numpy().astype(np.float64)
    if spheres_np.ndim == 4:  # [B, H, N, 4]
        spheres = spheres_np[0, 0]
    elif spheres_np.ndim == 3:  # [B, N, 4]
        spheres = spheres_np[0]
    else:
        raise RuntimeError(f"Unexpected sphere tensor shape: {spheres_np.shape}")
    centers = spheres[:, :3]
    radii = spheres[:, 3]

    sc_cfg = motion_gen.kinematics.get_self_collision_config()
    offsets = sc_cfg.offset.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if offsets.shape[0] != spheres.shape[0]:
        offsets = np.zeros((spheres.shape[0],), dtype=np.float64)

    sphere_links = _build_sphere_link_names(motion_gen)
    ignore_map = robot_cfg.get("robot_cfg", {}).get("kinematics", {}).get("self_collision_ignore", {})

    out: List[Tuple[float, Dict[str, Any]]] = []
    n = spheres.shape[0]
    for i in range(n - 1):
        if radii[i] <= 0.0:
            continue
        li = sphere_links[i]
        for j in range(i + 1, n):
            if radii[j] <= 0.0:
                continue
            lj = sphere_links[j]
            if _is_ignored_pair(li, lj, ignore_map):
                continue
            threshold = radii[i] + radii[j] + offsets[i] + offsets[j]
            if threshold <= 0.0:
                continue
            dist = float(np.linalg.norm(centers[i] - centers[j]))
            penetration = float(threshold - dist)
            if penetration > 0.0:
                out.append(
                    (
                        penetration,
                        {
                            "link_a": li,
                            "link_b": lj,
                            "sphere_a": int(i),
                            "sphere_b": int(j),
                            "distance": dist,
                            "threshold": float(threshold),
                            "penetration": penetration,
                        },
                    )
                )
    out.sort(key=lambda x: x[0], reverse=True)
    if int(max_pairs) <= 0:
        return [x[1] for x in out]
    return [x[1] for x in out[: max(1, int(max_pairs))]]


def _run_motiongen_plan(
    robot_cfg: Dict[str, Any],
    world_dict: Dict[str, Any],
    start_qpos: np.ndarray,
    target_pose_4x4: np.ndarray,
    seeds: int,
    self_collision: bool,
    debug_self_collision_pairs: bool = False,
    max_self_collision_pairs: int = 20,
) -> Dict[str, Any]:
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
        self_collision_check=self_collision,
        self_collision_opt=self_collision,
        store_ik_debug=True,
        store_debug_in_result=True,
    )
    motion_gen = MotionGen(mg_cfg)

    active_joint_names = list(getattr(motion_gen.kinematics, "joint_names", []))
    active_dof = len(active_joint_names)
    sq = np.asarray(start_qpos, dtype=np.float32).reshape(-1)
    if active_dof == 0:
        active_dof = sq.shape[0]
    if sq.shape[0] != active_dof:
        sq = _select_active_qpos_from_cspace(sq, robot_cfg, active_joint_names)
        if sq.shape[0] != active_dof:
            raise ValueError(f"start_qpos dof mismatch: start={sq.shape[0]}, active={active_dof}")

    start = JointState.from_position(
        torch.tensor(sq, device=tensor_args.device, dtype=torch.float32),
        joint_names=active_joint_names if len(active_joint_names) > 0 else None,
    ).unsqueeze(0)
    goal = Pose(
        position=torch.tensor(target_pose_4x4[:3, 3], device=tensor_args.device, dtype=torch.float32).view(1, 3),
        quaternion=torch.tensor(_rotmat_to_wxyz(target_pose_4x4[:3, :3]), device=tensor_args.device, dtype=torch.float32).view(1, 4),
    )
    plan_cfg = MotionGenPlanConfig(
        enable_graph=False,
        max_attempts=20,
        enable_graph_attempt=3,
        enable_finetune_trajopt=True,
        num_ik_seeds=int(seeds),
        num_trajopt_seeds=64,
    )
    result = motion_gen.plan_single(start, goal, plan_cfg)
    success = bool(result.success.item()) if hasattr(result.success, "item") else bool(result.success)

    out = {
        "mode": "motiongen",
        "success": success,
        "status": str(result.status),
        "active_dof": int(active_dof),
        "active_joint_names": active_joint_names,
        "trajectory_active": None,
    }
    if self_collision and debug_self_collision_pairs:
        pairs = _detect_self_collision_pairs(
            motion_gen=motion_gen,
            state=start,
            robot_cfg=robot_cfg,
            max_pairs=max_self_collision_pairs,
        )
        out["start_self_collision_count"] = len(pairs)
        out["start_self_collision_pairs"] = pairs

    if success:
        interp = result.get_interpolated_plan()
        out["trajectory_active"] = interp.position.detach().cpu().numpy().astype(np.float32)
    else:
        dbg = getattr(result, "debug_info", None)
        ik_result = dbg.get("ik_result") if isinstance(dbg, dict) else None
        if ik_result is not None:
            succ = ik_result.success.detach().cpu().reshape(-1).numpy().astype(bool)
            pe = ik_result.position_error.detach().cpu().reshape(-1).numpy().astype(np.float64)
            re = ik_result.rotation_error.detach().cpu().reshape(-1).numpy().astype(np.float64)
            score = pe + re
            best_idx = int(np.argmin(score))
            sol = ik_result.solution.detach().cpu().numpy().astype(np.float32).reshape(-1, active_dof)
            out["success_count"] = int(np.sum(succ))
            out["best_position_error"] = float(pe[best_idx])
            out["best_rotation_error"] = float(re[best_idx])
            out["best_qpos"] = sol[best_idx].tolist()
            if self_collision and debug_self_collision_pairs:
                best_state = JointState.from_position(
                    torch.tensor(sol[best_idx], device=tensor_args.device, dtype=torch.float32),
                    joint_names=active_joint_names if len(active_joint_names) > 0 else None,
                ).unsqueeze(0)
                best_pairs = _detect_self_collision_pairs(
                    motion_gen=motion_gen,
                    state=best_state,
                    robot_cfg=robot_cfg,
                    max_pairs=max_self_collision_pairs,
                )
                out["best_self_collision_count"] = len(best_pairs)
                out["best_self_collision_pairs"] = best_pairs
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planning_scene", required=True, help="path to planning_scene_curobo.json")
    parser.add_argument(
        "--robot_cfg",
        default=str(PROJECT_ROOT / "rsc" / "curobo" / "content" / "configs" / "robot" / "xarm_inspire_f1.yml"),
        help="path to curobo robot yml",
    )
    parser.add_argument("--ee_link", default=None, help="optional override ee_link in robot cfg")
    parser.add_argument("--seeds", type=int, default=256)
    parser.add_argument("--ignore_world", action="store_true")
    parser.add_argument("--arm_only", action="store_true")
    parser.add_argument("--start_qpos", default=None, help="optional csv start qpos override")
    parser.add_argument(
        "--lock_joints",
        action="store_true",
        help="lock hand/finger joints to start qpos via robot_cfg.kinematics.lock_joints",
    )
    parser.add_argument(
        "--hand_qpos_config_cost",
        action="store_true",
        help="apply hand-only null-space config cost around target hand qpos (pose target is kept)",
    )
    parser.add_argument(
        "--hand_qpos_weight",
        type=float,
        default=1.0,
        help="null-space weight for hand joints when --hand_qpos_config_cost is enabled",
    )
    parser.add_argument(
        "--hand_qpos_target",
        default=None,
        help="optional csv full-dof qpos target used as retract config (default: start_qpos)",
    )
    parser.add_argument(
        "--print_self_collision_pairs",
        action="store_true",
        help="print colliding link/sphere pairs at start state (for self_collision=True run)",
    )
    parser.add_argument(
        "--max_self_collision_pairs",
        type=int,
        default=0,
        help="max number of self-collision pairs to print (<=0 prints all)",
    )
    parser.add_argument("--vis", action="store_true", help="open viser and show start/ik/target/world")
    args = parser.parse_args()

    with open(args.planning_scene, "r", encoding="utf-8") as f:
        scene = json.load(f)
    target_pose = np.asarray(scene["target"]["target_eef_pose_robot"], dtype=np.float32)
    start_qpos = np.asarray(scene["target"].get("start_qpos", []), dtype=np.float32).reshape(-1)
    if args.start_qpos is not None:
        start_qpos = _parse_qpos(args.start_qpos, "--start_qpos")
    if start_qpos.size == 0:
        start_qpos = np.zeros(12, dtype=np.float32)
        n_arm = min(6, start_qpos.shape[0])
        start_qpos[:n_arm] = DEFAULT_INIT_ARM_QPOS[:n_arm]

    robot_cfg_path = os.path.expanduser(args.robot_cfg)
    robot_cfg = load_yaml(robot_cfg_path)
    robot_cfg = _normalize_robot_cfg_paths(robot_cfg, robot_cfg_path)
    if args.ee_link:
        robot_cfg["robot_cfg"]["kinematics"]["ee_link"] = str(args.ee_link)

    start_qpos = _resolve_start_qpos(start_qpos, robot_cfg)
    cspace_names = robot_cfg["robot_cfg"]["kinematics"].get("cspace", {}).get("joint_names", [])
    if args.arm_only:
        robot_cfg = _make_arm_only_robot_cfg(robot_cfg, start_qpos, list(cspace_names), arm_dof=6)
        start_qpos = _resolve_start_qpos(start_qpos, robot_cfg)
    if args.lock_joints and not args.arm_only:
        robot_cfg = _lock_hand_joints(robot_cfg, start_qpos, arm_dof=6)
    if args.hand_qpos_config_cost and not args.arm_only:
        hand_target = start_qpos
        if args.hand_qpos_target is not None:
            hand_target = _parse_qpos(args.hand_qpos_target, "--hand_qpos_target")
            hand_target = _resolve_start_qpos(hand_target, robot_cfg)
        robot_cfg = _apply_hand_qpos_config_cost(
            robot_cfg=robot_cfg,
            target_qpos=hand_target,
            hand_weight=float(args.hand_qpos_weight),
            arm_dof=6,
        )
    elif args.hand_qpos_config_cost and args.arm_only:
        print("warning: --hand_qpos_config_cost ignored because --arm_only locks/removes hand joints.")

    world_dict = {"cuboid": {}, "mesh": {}} if args.ignore_world else scene.get("world", {"cuboid": {}, "mesh": {}})
    world_dict = _sanitize_world_dict(world_dict)

    print(f"planning_scene: {args.planning_scene}")
    print(f"robot_cfg: {robot_cfg_path}")
    print(f"ee_link: {robot_cfg['robot_cfg']['kinematics'].get('ee_link')}")
    print(f"target position: {target_pose[:3, 3].tolist()}")
    print(f"target quaternion(wxyz): {_rotmat_to_wxyz(target_pose[:3, :3]).tolist()}")
    print(f"start_qpos_dof: {int(start_qpos.shape[0])}")
    if args.lock_joints and not args.arm_only:
        print("lock_joints: enabled (hand joints locked to start_qpos)")
    if args.hand_qpos_config_cost and not args.arm_only:
        print(f"hand_qpos_config_cost: enabled (hand_weight={float(args.hand_qpos_weight):.4f})")

    for self_collision in (False, True):
        out = _run_motiongen_plan(
            robot_cfg=robot_cfg,
            world_dict=world_dict,
            start_qpos=start_qpos,
            target_pose_4x4=target_pose,
            seeds=int(args.seeds),
            self_collision=self_collision,
            debug_self_collision_pairs=bool(args.print_self_collision_pairs),
            max_self_collision_pairs=int(args.max_self_collision_pairs),
        )
        msg = f"self_collision={self_collision} | success={out['success']} | status={out.get('status')} | active_dof={out['active_dof']}"
        if "best_position_error" in out:
            msg += (
                f" | best_pos_err={out['best_position_error']:.6f}"
                f" | best_rot_err={out['best_rotation_error']:.6f}"
                f" | success_count={out.get('success_count', 0)}"
            )
        print(msg)
        if self_collision and args.print_self_collision_pairs:
            pairs = out.get("start_self_collision_pairs", [])
            print(f"start_self_collision_pairs: {len(pairs)}")
            for i, p in enumerate(pairs):
                print(
                    f"  [{i:02d}] {p['link_a']}#{p['sphere_a']} <-> {p['link_b']}#{p['sphere_b']}"
                    f" | penetration={p['penetration']:.6f}"
                    f" | dist={p['distance']:.6f} | thr={p['threshold']:.6f}"
                )
            best_pairs = out.get("best_self_collision_pairs", [])
            if len(best_pairs) > 0:
                print(f"best_self_collision_pairs: {len(best_pairs)}")
                for i, p in enumerate(best_pairs):
                    print(
                        f"  [{i:02d}] {p['link_a']}#{p['sphere_a']} <-> {p['link_b']}#{p['sphere_b']}"
                        f" | penetration={p['penetration']:.6f}"
                        f" | dist={p['distance']:.6f} | thr={p['threshold']:.6f}"
                    )
        if out.get("best_qpos") is not None:
            print("best_qpos:", out["best_qpos"])

    if args.vis:
        urdf_path = robot_cfg["robot_cfg"]["kinematics"]["urdf_path"]
        vis = ViserViewer()
        vis.add_floor(height=0.0)
        vis.add_robot("robot_start", urdf_path)
        vis_joint_names = []
        if "out" in locals() and isinstance(out, dict):
            vis_joint_names = list(out.get("active_joint_names", []))
        if len(vis_joint_names) == start_qpos.shape[0]:
            vis.robot_dict["robot_start"].update_cfg(_qpos_to_cfg_map(vis_joint_names, start_qpos))
        else:
            vis.robot_dict["robot_start"].update_cfg(start_qpos)

        if "out" in locals() and isinstance(out, dict) and out.get("trajectory_active") is not None:
            traj_active = np.asarray(out["trajectory_active"], dtype=np.float32)
            if traj_active.shape[-1] == start_qpos.shape[0]:
                vis.add_traj("motiongen_traj", {"robot_start": traj_active}, {})

        # show best IK from the latest run (self_collision=True pass), when motiongen failed
        if "out" in locals() and isinstance(out, dict) and out.get("best_qpos") is not None:
            best_q = np.asarray(out["best_qpos"], dtype=np.float32).reshape(-1)
            if best_q.shape[0] == start_qpos.shape[0]:
                vis.add_robot("robot_ik_best", urdf_path)
                best_joint_names = list(out.get("active_joint_names", []))
                if len(best_joint_names) == best_q.shape[0]:
                    vis.robot_dict["robot_ik_best"].update_cfg(_qpos_to_cfg_map(best_joint_names, best_q))
                else:
                    vis.robot_dict["robot_ik_best"].update_cfg(best_q)

        target_q = _rotmat_to_wxyz(target_pose[:3, :3])
        vis.server.scene.add_frame(
            "/target/eef_pose",
            position=target_pose[:3, 3].astype(float),
            wxyz=target_q.astype(float),
            show_axes=True,
            axes_length=0.08,
            axes_radius=0.004,
        )

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
