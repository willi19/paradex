from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[2]))
PROJECT_ROOT = Path(__file__).parents[2]

import argparse
import json
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from paradex.visualization.robot import RobotModule


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"json object expected: {path}")
    return obj


def _expand_arm_qpos_to_full(arm_qpos: np.ndarray, full_seed_qpos: np.ndarray) -> np.ndarray:
    out = np.asarray(full_seed_qpos, dtype=np.float32).copy()
    arm = np.asarray(arm_qpos, dtype=np.float32).reshape(-1)
    n = min(arm.shape[0], out.shape[0], 6)
    out[:n] = arm[:n]
    return out


def _expand_active_qpos_to_full(
    active_qpos: np.ndarray,
    full_seed_qpos: np.ndarray,
    full_joint_names: List[str],
    active_joint_names: List[str],
) -> np.ndarray:
    q = np.asarray(active_qpos, dtype=np.float32).reshape(-1)
    out = np.asarray(full_seed_qpos, dtype=np.float32).copy()
    if q.shape[0] == out.shape[0]:
        return q
    if len(active_joint_names) != q.shape[0]:
        raise ValueError(f"active qpos/name mismatch: q={q.shape[0]}, active_names={len(active_joint_names)}")
    idx_map = {str(jn): i for i, jn in enumerate(full_joint_names)}
    for i, jn in enumerate(active_joint_names):
        idx = idx_map.get(str(jn))
        if idx is None:
            raise ValueError(f"active joint '{jn}' not found in full joint names")
        out[idx] = float(q[i])
    return out


def _build_full_trajectory(
    curobo_result: Dict[str, Any],
    start_qpos: np.ndarray,
    full_joint_names: List[str],
) -> np.ndarray:
    if curobo_result.get("trajectory_full") is not None:
        traj = np.asarray(curobo_result["trajectory_full"], dtype=np.float32)
        if traj.ndim == 2 and traj.shape[0] > 0:
            return traj
    if curobo_result.get("trajectory") is None:
        raise ValueError("curobo_result has no trajectory")
    traj = np.asarray(curobo_result["trajectory"], dtype=np.float32)
    if traj.ndim != 2 or traj.shape[0] == 0:
        raise ValueError("trajectory is empty/invalid")
    if traj.shape[1] == start_qpos.shape[0]:
        return traj
    if traj.shape[1] == 6:
        return np.stack([_expand_arm_qpos_to_full(q, start_qpos) for q in traj], axis=0)
    active_joint_names = list(curobo_result.get("active_joint_names", []) or [])
    return np.stack(
        [_expand_active_qpos_to_full(q, start_qpos, full_joint_names, active_joint_names) for q in traj],
        axis=0,
    )


def _make_scene_xml(
    robot_mjcf_path: str,
    world_dict: Dict[str, Any],
    actuator_joint_names: List[str],
    arm_kp: float,
    arm_kd: float,
    arm_force: float,
    hand_kp: float,
    hand_kd: float,
    hand_force: float,
    thumb_kp: float,
    thumb_kd: float,
    thumb_force: float,
    object_mass: float,
    use_xarm7_arm_profile: bool,
    use_existing_actuator_if_present: bool,
) -> str:
    robot_mjcf_abs = os.path.abspath(os.path.expanduser(robot_mjcf_path))
    with open(robot_mjcf_abs, "r", encoding="utf-8") as f:
        xml = f.read()
    base_dir = os.path.dirname(robot_mjcf_abs)

    def _replace_file_attr(match) -> str:
        src = match.group(1)
        if os.path.isabs(src):
            return f'file="{src}"'
        return f'file="{os.path.abspath(os.path.join(base_dir, src))}"'

    xml = re.sub(r'file="([^"]+)"', _replace_file_attr, xml)

    asset_inject: List[str] = []
    world_inject: List[str] = []
    table = world_dict.get("cuboid", {}).get("table")
    if isinstance(table, dict):
        dims = np.asarray(table.get("dims", [1.0, 1.0, 0.1]), dtype=np.float32).reshape(-1)
        pose = np.asarray(table.get("pose", [0, 0, -0.05, 1, 0, 0, 0]), dtype=np.float32).reshape(-1)
        sx, sy, sz = 0.5 * float(dims[0]), 0.5 * float(dims[1]), 0.5 * float(dims[2])
        px, py, pz = float(pose[0]), float(pose[1]), float(pose[2])
        world_inject.append(
            f'<geom name="scene_table" type="box" size="{sx} {sy} {sz}" pos="{px} {py} {pz}" '
            f'rgba="0.63 0.51 0.39 1" friction="1.0 0.01 0.001"/>'
        )
    obj = world_dict.get("mesh", {}).get("object")
    if isinstance(obj, dict):
        mesh_path = os.path.abspath(str(obj.get("file_path", "")))
        pose = np.asarray(obj.get("pose", [0, 0, 0, 1, 0, 0, 0]), dtype=np.float32).reshape(-1)
        if os.path.exists(mesh_path):
            x, y, z, qw, qx, qy, qz = [float(v) for v in pose[:7]]
            asset_inject.append(f'<mesh name="scene_object_mesh" file="{mesh_path}"/>')
            world_inject.append(
                f'<body name="scene_object" pos="{x} {y} {z}" quat="{qw} {qx} {qy} {qz}">'
                f'<freejoint/><geom name="scene_object_geom" type="mesh" mesh="scene_object_mesh" mass="{float(object_mass)}" '
                f'rgba="0.2 0.6 0.9 1" friction="1.0 0.03 0.001"/></body>'
            )
    if asset_inject:
        xml = xml.replace("</asset>", "  " + "\n  ".join(asset_inject) + "\n  </asset>")
    if world_inject:
        xml = xml.replace("</worldbody>", "  " + "\n  ".join(world_inject) + "\n  </worldbody>")

    def _arm_profile(jn: str) -> Tuple[float, float, float]:
        if use_xarm7_arm_profile and jn.startswith("joint"):
            m = re.match(r"joint(\d+)$", jn)
            idx = int(m.group(1)) if m else 6
            if idx <= 2:
                return 1700.0, 220.0, 80.0
            if idx <= 5:
                return 1200.0, 150.0, 50.0
            return 900.0, 120.0, 30.0
        return float(arm_kp), float(arm_kd), float(arm_force)

    def _inspire_hand_pd(jn: str) -> Optional[Tuple[float, float, float]]:
        s = str(jn)
        if "thumb_1_joint" in s:
            return 120.0, 18.0, 20.0
        if "thumb_2_joint" in s:
            return 100.0, 16.0, 16.0
        if any(k in s for k in ("index_1_joint", "middle_1_joint", "ring_1_joint", "little_1_joint")):
            return 80.0, 14.0, 12.0
        return None

    has_existing_actuator = "<actuator" in xml
    if not (use_existing_actuator_if_present and has_existing_actuator):
        dependent_joints = set(re.findall(r'joint1="([^"]+)"', xml))
        act_lines = ["<actuator>"]
        for jn in actuator_joint_names:
            if str(jn) in dependent_joints:
                continue
            if jn.startswith("joint"):
                kp, kd, force = _arm_profile(jn)
            else:
                hp = _inspire_hand_pd(str(jn))
                if hp is not None:
                    kp, kd, force = hp
                elif "thumb" in str(jn):
                    kp, kd, force = float(thumb_kp), float(thumb_kd), float(thumb_force)
                else:
                    kp, kd, force = float(hand_kp), float(hand_kd), float(hand_force)
            act_lines.append(
                f'  <general name="act_{jn}" joint="{jn}" biastype="affine" '
                f'gainprm="{kp}" biasprm="0 -{kp} -{kd}" forcerange="-{force} {force}" ctrllimited="false"/>'
            )
        act_lines.append("</actuator>")
        xml = xml.replace("</mujoco>", "  " + "\n  ".join(act_lines) + "\n</mujoco>")

    fd, out_path = tempfile.mkstemp(prefix="mujoco_hand_pd_tune_", suffix=".xml")
    os.close(fd)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    return out_path


def _run_replay(
    scene: Dict[str, Any],
    traj: np.ndarray,
    full_joint_names: List[str],
    robot_mjcf_path: str,
    arm_kp: float,
    arm_kd: float,
    arm_force: float,
    hand_kp: float,
    hand_kd: float,
    hand_force: float,
    hand_ctrl_max_step: float,
    thumb_kp: float,
    thumb_kd: float,
    thumb_force: float,
    thumb_ctrl_max_step: float,
    object_mass: float,
    waypoint_hold_steps: int,
    start_settle_steps: int,
    decimation: int,
    lift_success_z: float,
    use_xarm7_arm_profile: bool,
    use_existing_actuator_if_present: bool,
) -> None:
    import mujoco
    import mujoco.viewer

    # Probe mapping first.
    probe_model = mujoco.MjModel.from_xml_path(os.path.abspath(os.path.expanduser(robot_mjcf_path)))
    mapped_joint_names: List[str] = []
    for jn in full_joint_names:
        if mujoco.mj_name2id(probe_model, mujoco.mjtObj.mjOBJ_JOINT, str(jn)) >= 0:
            mapped_joint_names.append(str(jn))

    scene_xml = _make_scene_xml(
        robot_mjcf_path=robot_mjcf_path,
        world_dict=scene["world"],
        actuator_joint_names=mapped_joint_names,
        arm_kp=arm_kp,
        arm_kd=arm_kd,
        arm_force=arm_force,
        hand_kp=hand_kp,
        hand_kd=hand_kd,
        hand_force=hand_force,
        thumb_kp=thumb_kp,
        thumb_kd=thumb_kd,
        thumb_force=thumb_force,
        object_mass=object_mass,
        use_xarm7_arm_profile=use_xarm7_arm_profile,
        use_existing_actuator_if_present=use_existing_actuator_if_present,
    )
    with open(scene_xml, "r", encoding="utf-8") as f:
        _scene_xml_text = f.read()
    dependent_joints = set(re.findall(r'joint1="([^"]+)"', _scene_xml_text))
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)

    actuator_obj = mujoco.mjtObj.mjOBJ_ACTUATOR
    mapping: List[Tuple[int, int, int, bool, bool, float, float]] = []
    for src_idx, jn in enumerate(full_joint_names):
        if str(jn) in dependent_joints:
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(jn))
        if jid < 0:
            continue
        qpos_idx = int(model.jnt_qposadr[jid])
        aid = int(mujoco.mj_name2id(model, actuator_obj, f"act_{jn}"))
        if aid < 0:
            aid = int(mujoco.mj_name2id(model, actuator_obj, str(jn)))
        is_hand = (not str(jn).startswith("joint"))
        is_thumb = ("thumb" in str(jn))
        if int(model.jnt_limited[jid]) == 1:
            lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
        else:
            lo, hi = -1e6, 1e6
        mapping.append((src_idx, qpos_idx, aid, is_hand, is_thumb, lo, hi))

    print(f"mapped joints: {len(mapping)}/{len(full_joint_names)}, traj steps={traj.shape[0]}")
    print(f"simulate: timestep={float(model.opt.timestep):.6f}, decimation={int(decimation)}")
    if use_existing_actuator_if_present:
        print("hand PD: using MJCF actuator parameters (script kp/kd/force args are ignored)")
    else:
        print(
            "hand PD:",
            f"finger(kp,kd,F,step)=({hand_kp},{hand_kd},{hand_force},{hand_ctrl_max_step}),",
            f"thumb(kp,kd,F,step)=({thumb_kp},{thumb_kd},{thumb_force},{thumb_ctrl_max_step})",
        )

    for src_idx, qpos_idx, _aid, _is_hand, _is_thumb, lo, hi in mapping:
        data.qpos[qpos_idx] = float(np.clip(float(traj[0, src_idx]), lo, hi))
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "scene_object")
    obj_start_z = float(data.xpos[obj_bid, 2]) if obj_bid >= 0 else None
    obj_max_z = obj_start_z
    sim_decimation = max(1, int(decimation))

    def _step_sim(viewer) -> None:
        nonlocal obj_max_z
        # Follow RoboVerse MujocoHandler._simulate(): physics.step(decimation)
        for _ in range(sim_decimation):
            mujoco.mj_step(model, data)
            if obj_bid >= 0:
                obj_max_z = max(float(obj_max_z), float(data.xpos[obj_bid, 2]))
            viewer.sync()
            time.sleep(model.opt.timestep)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Initialize ctrl once from first waypoint (set_dof_targets style).
        for src_idx, _qpos_idx, aid, is_hand, is_thumb, lo, hi in mapping:
            if not (0 <= aid < model.nu):
                continue
            target = float(np.clip(float(traj[0, src_idx]), lo, hi))
            prev = float(data.ctrl[aid])
            if is_hand and hand_ctrl_max_step > 0.0:
                ms = float(thumb_ctrl_max_step if is_thumb else hand_ctrl_max_step)
                target = prev + float(np.clip(target - prev, -ms, ms))
            data.ctrl[aid] = target

        # settle
        for _ in range(max(0, int(start_settle_steps))):
            _step_sim(viewer)

        for k in range(1, traj.shape[0]):
            q = traj[k]
            for _ in range(max(1, int(waypoint_hold_steps))):
                # One control update per control step (set_dof_targets style).
                for src_idx, _qpos_idx, aid, is_hand, is_thumb, lo, hi in mapping:
                    if src_idx >= q.shape[0] or not (0 <= aid < model.nu):
                        continue
                    target = float(np.clip(float(q[src_idx]), lo, hi))
                    prev = float(data.ctrl[aid])
                    if is_hand and hand_ctrl_max_step > 0.0:
                        ms = float(thumb_ctrl_max_step if is_thumb else hand_ctrl_max_step)
                        target = prev + float(np.clip(target - prev, -ms, ms))
                    data.ctrl[aid] = target
                _step_sim(viewer)

        if obj_start_z is not None and obj_max_z is not None:
            dz = float(obj_max_z - obj_start_z)
            print(
                f"Lift validation: start_z={obj_start_z:.4f}, max_z={obj_max_z:.4f}, "
                f"delta_z={dz:.4f}, threshold={float(lift_success_z):.4f}, success={dz >= float(lift_success_z)}"
            )
        print("Playback finished. Close MuJoCo window to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True, help="capture_rcc result directory (contains planning_scene_curobo.json/curobo_plan_result.json)")
    parser.add_argument("--planning_scene", default=None)
    parser.add_argument("--curobo_result", default=None)
    parser.add_argument(
        "--mujoco_robot_mjcf",
        default=str(PROJECT_ROOT / "rsc" / "curobo" / "content" / "assets" / "robot" / "inspire_description" / "xarm_inspire_f1.mjcf"),
    )
    parser.add_argument("--arm_kp", type=float, default=1200.0)
    parser.add_argument("--arm_kd", type=float, default=180.0)
    parser.add_argument("--arm_force", type=float, default=80.0)
    parser.add_argument("--hand_kp", type=float, default=80.0)
    parser.add_argument("--hand_kd", type=float, default=16.0)
    parser.add_argument("--hand_force", type=float, default=20.0)
    parser.add_argument("--hand_ctrl_max_step", type=float, default=0.02)
    parser.add_argument("--thumb_kp", type=float, default=120.0)
    parser.add_argument("--thumb_kd", type=float, default=24.0)
    parser.add_argument("--thumb_force", type=float, default=20.0)
    parser.add_argument("--thumb_ctrl_max_step", type=float, default=0.02)
    parser.add_argument("--start_settle_steps", type=int, default=180)
    parser.add_argument("--waypoint_hold_steps", type=int, default=3)
    parser.add_argument("--decimation", type=int, default=3)
    parser.add_argument("--object_mass", type=float, default=0.3)
    parser.add_argument("--lift_success_z", type=float, default=0.05)
    parser.add_argument("--no_xarm7_arm_profile", action="store_true")
    parser.add_argument(
        "--force_generated_actuator",
        action="store_true",
        help="ignore actuator block in MJCF and generate temporary general PD actuators from this script",
    )
    args = parser.parse_args()

    result_dir = os.path.abspath(os.path.expanduser(args.result_dir))
    planning_scene_path = os.path.abspath(os.path.expanduser(args.planning_scene or os.path.join(result_dir, "planning_scene_curobo.json")))
    curobo_result_path = os.path.abspath(os.path.expanduser(args.curobo_result or os.path.join(result_dir, "curobo_plan_result.json")))

    scene = _load_json(planning_scene_path)
    curobo_result = _load_json(curobo_result_path)

    robot_urdf = scene.get("robot", {}).get("urdf_path")
    if not robot_urdf or not os.path.exists(robot_urdf):
        raise FileNotFoundError(f"invalid robot urdf_path in planning scene: {robot_urdf}")
    robot_module = RobotModule(robot_urdf)
    full_joint_names = list(robot_module.get_joint_names())
    start_qpos = np.asarray(scene.get("target", {}).get("start_qpos", []), dtype=np.float32).reshape(-1)
    if start_qpos.shape[0] != len(full_joint_names):
        raise ValueError(f"start_qpos dof mismatch: start={start_qpos.shape[0]}, joints={len(full_joint_names)}")

    traj = _build_full_trajectory(curobo_result=curobo_result, start_qpos=start_qpos, full_joint_names=full_joint_names)
    _run_replay(
        scene=scene,
        traj=traj,
        full_joint_names=full_joint_names,
        robot_mjcf_path=args.mujoco_robot_mjcf,
        arm_kp=float(args.arm_kp),
        arm_kd=float(args.arm_kd),
        arm_force=float(args.arm_force),
        hand_kp=float(args.hand_kp),
        hand_kd=float(args.hand_kd),
        hand_force=float(args.hand_force),
        hand_ctrl_max_step=float(args.hand_ctrl_max_step),
        thumb_kp=float(args.thumb_kp),
        thumb_kd=float(args.thumb_kd),
        thumb_force=float(args.thumb_force),
        thumb_ctrl_max_step=float(args.thumb_ctrl_max_step),
        object_mass=float(args.object_mass),
        waypoint_hold_steps=int(args.waypoint_hold_steps),
        start_settle_steps=int(args.start_settle_steps),
        decimation=int(args.decimation),
        lift_success_z=float(args.lift_success_z),
        use_xarm7_arm_profile=(not bool(args.no_xarm7_arm_profile)),
        use_existing_actuator_if_present=(not bool(args.force_generated_actuator)),
    )


if __name__ == "__main__":
    main()
