from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[2]))
PROJECT_ROOT = Path(__file__).parents[2]

import argparse
import datetime
import json
import os
import re
import tempfile
import time
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import trimesh
import zmq

from paradex.calibration.utils import load_current_C2R, save_current_camparam
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.utils.path import shared_dir
from paradex.visualization.robot import RobotModule
from paradex.visualization.visualizer.viser import ViserViewer

DEFAULT_ROBOT_URDF = str(PROJECT_ROOT / "rsc" / "curobo" / "content" / "assets" / "robot" / "inspire_description" / "xarm_inspire_f1.urdf")
DEFAULT_CUROBO_ROBOT_CFG = str(PROJECT_ROOT / "rsc" / "curobo" / "content" / "configs" / "robot" / "xarm_inspire_f1.yml")
DEFAULT_DEMO_BASE = "/home/temp_id/shared_data/capture/eccv2026/inspire_f1"
DEFAULT_INIT_ARM_QPOS = np.array(
    [-1.481784, -0.445059, -0.71733, 2.073451, -0.033161, 2.759586],
    dtype=np.float32,
)


def _send_rpc_once(addr: str, req: dict, timeout_ms: int = 300000) -> dict:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.connect(addr)
        sock.send_json(req)
        return sock.recv_json()
    finally:
        sock.close()
        ctx.term()


def _to_shared_data_path(abs_path: str) -> str:
    norm = os.path.normpath(abs_path)
    marker = f"{os.sep}shared_data{os.sep}"
    if marker in norm:
        tail = norm.split(marker, 1)[1]
        return os.path.join("shared_data", tail).replace(os.sep, "/")
    if norm.endswith(f"{os.sep}shared_data"):
        return "shared_data"
    raise ValueError(f"path is not under shared_data: {abs_path}")


def _parse_cam_from_world(extrinsic_value: Any) -> np.ndarray:
    arr = np.asarray(extrinsic_value, dtype=float)
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=float)
        out[:3, :] = arr
        return out
    if arr.shape == (4, 4):
        return arr
    if arr.size == 12:
        out = np.eye(4, dtype=float)
        out[:3, :] = arr.reshape(3, 4)
        return out
    raise ValueError(f"Unsupported extrinsic shape: {arr.shape} (expected 3x4 or 4x4)")


def _build_intrinsic_for_viewer(serial: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if "intrinsics_undistort" in payload:
        K = np.asarray(payload["intrinsics_undistort"], dtype=float)
    elif "original_intrinsics" in payload:
        K = np.asarray(payload["original_intrinsics"], dtype=float)
    else:
        raise ValueError(f"{serial}: missing intrinsic fields")

    width = payload.get("width")
    height = payload.get("height")
    if width is None or height is None:
        width = int(round(float(K[0, 2]) * 2.0))
        height = int(round(float(K[1, 2]) * 2.0))

    return {
        "intrinsics_undistort": K.tolist(),
        "width": int(width),
        "height": int(height),
    }


def _to_4x4(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape == (4, 4):
        return mat
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :] = mat
        return out
    raise ValueError(f"matrix must be 4x4 or 3x4, got {mat.shape}")


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


def _make_mujoco_scene_xml(
    robot_mjcf_path: str,
    world_dict: Dict[str, Any],
    object_mass: float = 0.05,
    actuator_joint_names: Optional[List[str]] = None,
    actuator_arm_only: bool = False,
    hand_use_position_actuator: bool = True,
    actuator_use_xarm7_profile: bool = True,
    actuator_kp_arm: float = 1200.0,
    actuator_kd_arm: float = 140.0,
    actuator_force_arm: float = 50.0,
    actuator_kp_hand: float = 80.0,
    actuator_kd_hand: float = 16.0,
    actuator_force_hand: float = 15.0,
    hand_position_kp: float = 8.0,
    use_existing_actuator_if_present: bool = True,
) -> str:
    robot_mjcf_abs = os.path.abspath(os.path.expanduser(robot_mjcf_path))
    if not os.path.exists(robot_mjcf_abs):
        raise FileNotFoundError(f"mujoco robot mjcf not found: {robot_mjcf_abs}")
    mjcf_dir = os.path.dirname(robot_mjcf_abs)
    with open(robot_mjcf_abs, "r", encoding="utf-8") as f:
        xml = f.read()

    # Normalize mesh file paths to absolute paths so loading is stable regardless of cwd.
    def _replace_file_attr(match) -> str:
        src = match.group(1)
        if os.path.isabs(src):
            return f'file="{src}"'
        return f'file="{os.path.abspath(os.path.join(mjcf_dir, src))}"'

    xml = re.sub(r'file="([^"]+)"', _replace_file_attr, xml)

    asset_inject: List[str] = []
    world_inject: List[str] = []

    table = world_dict.get("cuboid", {}).get("table")
    if isinstance(table, dict):
        dims = np.asarray(table.get("dims", [1.0, 1.0, 0.1]), dtype=np.float32).reshape(-1)
        pose = np.asarray(table.get("pose", [0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
        if dims.shape[0] >= 3 and pose.shape[0] >= 3:
            sx, sy, sz = 0.5 * float(dims[0]), 0.5 * float(dims[1]), 0.5 * float(dims[2])
            px, py, pz = float(pose[0]), float(pose[1]), float(pose[2])
            world_inject.append(
                f'<geom name="scene_table" type="box" size="{sx} {sy} {sz}" pos="{px} {py} {pz}" '
                f'rgba="0.63 0.51 0.39 1" friction="1.0 0.01 0.001"/>'
            )

    obj = world_dict.get("mesh", {}).get("object")
    if isinstance(obj, dict):
        mesh_path = os.path.abspath(str(obj.get("file_path", "")))
        pose = np.asarray(obj.get("pose", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
        if os.path.exists(mesh_path) and pose.shape[0] >= 7:
            x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
            qw, qx, qy, qz = float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])
            asset_inject.append(f'<mesh name="scene_object_mesh" file="{mesh_path}"/>')
            world_inject.append(
                f'<body name="scene_object" pos="{x} {y} {z}" quat="{qw} {qx} {qy} {qz}">'
                f'<freejoint/><geom name="scene_object_geom" type="mesh" mesh="scene_object_mesh" '
                f'mass="{float(object_mass)}" rgba="0.2 0.6 0.9 1" friction="1.0 0.03 0.001"/></body>'
            )

    if asset_inject:
        xml = xml.replace("</asset>", "  " + "\n  ".join(asset_inject) + "\n  </asset>")
    if world_inject:
        xml = xml.replace("</worldbody>", "  " + "\n  ".join(world_inject) + "\n  </worldbody>")

    if actuator_joint_names:
        has_existing_actuator = "<actuator" in xml
        if use_existing_actuator_if_present and has_existing_actuator:
            actuator_joint_names = None
    if actuator_joint_names:
        # If MJCF already has joint-equality constraints (mimic), actuate only source joints.
        dependent_joints = set(re.findall(r'joint1="([^"]+)"', xml))
        def _inspire_hand_pd(jn: str) -> Optional[Tuple[float, float, float]]:
            s = str(jn)
            if "thumb_1_joint" in s:
                return 120.0, 18.0, 20.0
            if "thumb_2_joint" in s:
                return 100.0, 16.0, 16.0
            if any(k in s for k in ("index_1_joint", "middle_1_joint", "ring_1_joint", "little_1_joint")):
                return 80.0, 14.0, 12.0
            return None
        def _arm_profile_by_joint_name(jn: str) -> Tuple[float, float, float]:
            if not str(jn).startswith("joint"):
                return float(actuator_kp_hand), float(actuator_kd_hand), float(actuator_force_hand)
            m = re.match(r"joint(\d+)$", str(jn))
            if not m:
                return float(actuator_kp_arm), float(actuator_kd_arm), float(actuator_force_arm)
            idx = int(m.group(1))
            if actuator_use_xarm7_profile:
                if idx <= 2:
                    return 1700.0, 220.0, 80.0
                if idx <= 5:
                    return 1200.0, 150.0, 50.0
                return 900.0, 120.0, 30.0
            return float(actuator_kp_arm), float(actuator_kd_arm), float(actuator_force_arm)

        act_lines: List[str] = ["<actuator>"]
        for jn in actuator_joint_names:
            if str(jn) in dependent_joints:
                continue
            if actuator_arm_only and (not str(jn).startswith("joint")):
                continue
            if hand_use_position_actuator and (not str(jn).startswith("joint")):
                # Shadow-hand style: position actuator with relatively small kp.
                force = float(actuator_force_hand)
                kp = float(hand_position_kp)
                act_lines.append(
                    f'  <position name="act_{jn}" joint="{jn}" kp="{kp}" '
                    f'forcelimited="true" forcerange="-{force} {force}" ctrllimited="false"/>'
                )
                continue
            if str(jn).startswith("joint"):
                kp, kd, force = _arm_profile_by_joint_name(str(jn))
            else:
                hp = _inspire_hand_pd(str(jn))
                if hp is not None:
                    kp, kd, force = hp
                else:
                    kp, kd, force = float(actuator_kp_hand), float(actuator_kd_hand), float(actuator_force_hand)
            act_lines.append(
                f'  <general name="act_{jn}" joint="{jn}" biastype="affine" '
                f'gainprm="{kp}" biasprm="0 -{kp} -{kd}" forcerange="-{force} {force}" ctrllimited="false"/>'
            )
        act_lines.append("</actuator>")
        xml = xml.replace("</mujoco>", "  " + "\n  ".join(act_lines) + "\n</mujoco>")

    fd, out_path = tempfile.mkstemp(prefix="capture_rcc_mj_", suffix=".xml")
    os.close(fd)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    return out_path


def _run_mujoco_replay(
    robot_mjcf_path: str,
    world_dict: Dict[str, Any],
    trajectory: np.ndarray,
    full_joint_names: List[str],
    waypoint_hold_steps: int = 4,
    object_mass: float = 0.05,
    dynamic_step: bool = False,
    validate_lift: bool = False,
    lift_success_z: float = 0.05,
    start_settle_steps: int = 60,
    hand_kinematic_in_dynamic: bool = False,
    hand_use_position_actuator: bool = True,
    actuator_use_xarm7_profile: bool = True,
    actuator_kp_arm: float = 1200.0,
    actuator_kd_arm: float = 140.0,
    actuator_force_arm: float = 50.0,
    actuator_kp_hand: float = 80.0,
    actuator_kd_hand: float = 16.0,
    actuator_force_hand: float = 15.0,
    hand_ctrl_max_step: float = 0.008,
    hand_position_kp: float = 8.0,
    use_existing_actuator_if_present: bool = True,
    decimation: int = 3,
) -> None:
    import mujoco
    import mujoco.viewer

    traj = np.asarray(trajectory, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[0] == 0:
        print("[WARN] empty trajectory, skip MuJoCo replay.")
        return

    # Build once to discover joint overlap, then rebuild with actuators if dynamic mode is requested.
    scene_xml_path = _make_mujoco_scene_xml(
        robot_mjcf_path=robot_mjcf_path,
        world_dict=world_dict,
        object_mass=float(object_mass),
    )
    with open(scene_xml_path, "r", encoding="utf-8") as f:
        _scene_xml_probe = f.read()
    dependent_joints = set(re.findall(r'joint1="([^"]+)"', _scene_xml_probe))
    probe_model = mujoco.MjModel.from_xml_path(scene_xml_path)
    mapping: List[Tuple[int, int]] = []
    mapped_joint_names: List[str] = []
    for src_idx, jn in enumerate(full_joint_names):
        if str(jn) in dependent_joints:
            continue
        joint_id = mujoco.mj_name2id(probe_model, mujoco.mjtObj.mjOBJ_JOINT, str(jn))
        if joint_id >= 0:
            mapping.append((src_idx, int(probe_model.jnt_qposadr[joint_id])))
            mapped_joint_names.append(str(jn))
    if not mapping:
        print("[WARN] no joint name overlap between curobo traj and MuJoCo model.")
        return
    if dynamic_step:
        scene_xml_path = _make_mujoco_scene_xml(
            robot_mjcf_path=robot_mjcf_path,
            world_dict=world_dict,
            object_mass=float(object_mass),
            actuator_joint_names=mapped_joint_names,
            actuator_arm_only=bool(hand_kinematic_in_dynamic),
            hand_use_position_actuator=bool(hand_use_position_actuator),
            actuator_use_xarm7_profile=bool(actuator_use_xarm7_profile),
            actuator_kp_arm=float(actuator_kp_arm),
            actuator_kd_arm=float(actuator_kd_arm),
            actuator_force_arm=float(actuator_force_arm),
            actuator_kp_hand=float(actuator_kp_hand),
            actuator_kd_hand=float(actuator_kd_hand),
            actuator_force_hand=float(actuator_force_hand),
            hand_position_kp=float(hand_position_kp),
            use_existing_actuator_if_present=bool(use_existing_actuator_if_present),
        )
    model = mujoco.MjModel.from_xml_path(scene_xml_path)
    data = mujoco.MjData(model)
    actuator_obj = mujoco.mjtObj.mjOBJ_ACTUATOR
    map_joint_ids: List[int] = []
    map_dof_adr: List[int] = []
    map_act_id: List[int] = []
    map_is_hand: List[bool] = []
    map_ctrl_lo: List[float] = []
    map_ctrl_hi: List[float] = []
    # Recompute joint ids/ranges against final model for safe ctrl clamping.
    for src_idx, _qpos_idx in mapping:
        jn = str(full_joint_names[src_idx]) if src_idx < len(full_joint_names) else ""
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        map_joint_ids.append(int(jid))
        map_dof_adr.append(int(model.jnt_dofadr[jid]) if jid >= 0 else -1)
        aid = mujoco.mj_name2id(model, actuator_obj, f"act_{jn}")
        if aid < 0:
            aid = mujoco.mj_name2id(model, actuator_obj, str(jn))
        map_act_id.append(int(aid))
        is_hand = (not jn.startswith("joint"))
        map_is_hand.append(is_hand)
        if jid >= 0 and int(model.jnt_limited[jid]) == 1:
            map_ctrl_lo.append(float(model.jnt_range[jid, 0]))
            map_ctrl_hi.append(float(model.jnt_range[jid, 1]))
        else:
            map_ctrl_lo.append(-1.0e6)
            map_ctrl_hi.append(1.0e6)

    mode = "dynamic-servo" if dynamic_step else "kinematic"
    print(f"MuJoCo replay ({mode}): mapped joints {len(mapping)}/{len(full_joint_names)}, traj steps={traj.shape[0]}")
    if dynamic_step:
        # Match xArm examples: implicit integrator + smaller timestep.
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        model.opt.timestep = 0.002
        has_generated_prefix = False
        for aid in range(int(model.nu)):
            aname = mujoco.mj_id2name(model, actuator_obj, aid)
            if isinstance(aname, str) and aname.startswith("act_"):
                has_generated_prefix = True
                break
        if has_generated_prefix:
            print(
                "MuJoCo servo gains (generated):",
                f"use_xarm7_profile={bool(actuator_use_xarm7_profile)}, "
                f"arm(kp,kd,F)=({float(actuator_kp_arm):.3f},{float(actuator_kd_arm):.3f},{float(actuator_force_arm):.3f}), "
                f"hand(kp,kd,F)=({float(actuator_kp_hand):.3f},{float(actuator_kd_hand):.3f},{float(actuator_force_hand):.3f})",
            )
            # Joint damping profile from xArm7 MJCF defaults.
            for jn in mapped_joint_names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(jn))
                if jid < 0:
                    continue
                dof_adr = int(model.jnt_dofadr[jid])
                if not str(jn).startswith("joint"):
                    model.dof_damping[dof_adr] = 2.0
                    continue
                m = re.match(r"joint(\d+)$", str(jn))
                idx = int(m.group(1)) if m else 6
                if idx <= 2:
                    model.dof_damping[dof_adr] = 16.0
                elif idx <= 5:
                    model.dof_damping[dof_adr] = 10.0
                else:
                    model.dof_damping[dof_adr] = 6.0
        else:
            print("MuJoCo servo gains: using actuator PD from MJCF")
        print(f"MuJoCo simulate: timestep={float(model.opt.timestep):.6f}, decimation={int(max(1, int(decimation)))}")
    if not dynamic_step:
        model.opt.timestep = 1.0 / 240.0
    data.qvel[:] = 0.0
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "scene_object")
    obj_start_z = None
    obj_max_z = None
    sim_decimation = max(1, int(decimation))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # initialize at first waypoint
        q0 = traj[0]
        for src_idx, qpos_idx in mapping:
            if src_idx < q0.shape[0]:
                data.qpos[qpos_idx] = float(q0[src_idx])
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if obj_bid >= 0:
            obj_start_z = float(data.xpos[obj_bid, 2])
            obj_max_z = obj_start_z

        def _set_control_targets(q_cmd: np.ndarray) -> None:
            for ai, (src_idx, qpos_idx) in enumerate(mapping):
                if src_idx >= q_cmd.shape[0]:
                    continue
                target = float(np.clip(float(q_cmd[src_idx]), map_ctrl_lo[ai], map_ctrl_hi[ai]))
                if hand_kinematic_in_dynamic and map_is_hand[ai]:
                    data.qpos[qpos_idx] = target
                    dof_idx = map_dof_adr[ai]
                    if 0 <= dof_idx < data.qvel.shape[0]:
                        data.qvel[dof_idx] = 0.0
                    continue
                aid = map_act_id[ai]
                if 0 <= aid < model.nu:
                    if map_is_hand[ai]:
                        prev = float(data.ctrl[aid])
                        max_step = float(max(0.0, hand_ctrl_max_step))
                        if max_step > 0.0:
                            target = prev + float(np.clip(target - prev, -max_step, max_step))
                    data.ctrl[aid] = target

        def _physics_step_block(q_cmd_for_hand: np.ndarray) -> None:
            nonlocal obj_max_z
            for _ in range(sim_decimation):
                if hand_kinematic_in_dynamic:
                    for ai, (src_idx, qpos_idx) in enumerate(mapping):
                        if src_idx < q_cmd_for_hand.shape[0] and map_is_hand[ai]:
                            target = float(np.clip(float(q_cmd_for_hand[src_idx]), map_ctrl_lo[ai], map_ctrl_hi[ai]))
                            data.qpos[qpos_idx] = target
                            dof_idx = map_dof_adr[ai]
                            if 0 <= dof_idx < data.qvel.shape[0]:
                                data.qvel[dof_idx] = 0.0
                mujoco.mj_step(model, data)
                if obj_bid >= 0:
                    oz = float(data.xpos[obj_bid, 2])
                    obj_max_z = oz if obj_max_z is None else max(obj_max_z, oz)
                viewer.sync()
                time.sleep(model.opt.timestep)

        start_idx = 0
        if dynamic_step and model.nu > 0:
            # set_dof_targets + simulate(decimation) style (same as mujoco_hand_pd_tune / rrc handler)
            _set_control_targets(q0)
            for _ in range(max(0, int(start_settle_steps))):
                _physics_step_block(q0)
            start_idx = 1

        for k in range(start_idx, traj.shape[0]):
            q = traj[k]
            for _ in range(max(1, int(waypoint_hold_steps))):
                if dynamic_step:
                    if model.nu <= 0:
                        print("[WARN] no actuators found; fallback to kinematic forward.")
                        for src_idx, qpos_idx in mapping:
                            if src_idx < q.shape[0]:
                                data.qpos[qpos_idx] = float(q[src_idx])
                        data.qvel[:] = 0.0
                        mujoco.mj_forward(model, data)
                        if obj_bid >= 0:
                            oz = float(data.xpos[obj_bid, 2])
                            obj_max_z = oz if obj_max_z is None else max(obj_max_z, oz)
                        viewer.sync()
                        time.sleep(model.opt.timestep)
                    else:
                        _set_control_targets(q)
                        _physics_step_block(q)
                else:
                    # Stable playback: kinematic forward only (no dynamics integration).
                    for src_idx, qpos_idx in mapping:
                        if src_idx < q.shape[0]:
                            data.qpos[qpos_idx] = float(q[src_idx])
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                    if obj_bid >= 0:
                        oz = float(data.xpos[obj_bid, 2])
                        obj_max_z = oz if obj_max_z is None else max(obj_max_z, oz)
                    viewer.sync()
                    time.sleep(model.opt.timestep)
        print("Trajectory playback finished. Close MuJoCo window to exit.")
        if validate_lift and obj_bid >= 0 and obj_start_z is not None and obj_max_z is not None:
            dz = float(obj_max_z - obj_start_z)
            print(
                f"Lift validation: start_z={obj_start_z:.4f}, max_z={obj_max_z:.4f}, "
                f"delta_z={dz:.4f}, threshold={float(lift_success_z):.4f}, success={dz >= float(lift_success_z)}"
            )
        while viewer.is_running():
            if dynamic_step:
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)
            else:
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)


def _add_cameras_to_viser(vis: ViserViewer, capture_root: str, c2r: np.ndarray) -> None:
    camparam_dir = os.path.join(capture_root, "cam_param")
    intr_path = os.path.join(camparam_dir, "intrinsics.json")
    extr_path = os.path.join(camparam_dir, "extrinsics.json")
    if not (os.path.exists(intr_path) and os.path.exists(extr_path)):
        print(f"[WARN] cam_param missing under {camparam_dir}")
        return

    with open(intr_path, "r", encoding="utf-8") as f:
        intrinsics = json.load(f)
    with open(extr_path, "r", encoding="utf-8") as f:
        extrinsics = json.load(f)

    serials = sorted(set(intrinsics.keys()) & set(extrinsics.keys()))
    for serial in serials:
        intrinsic_view = _build_intrinsic_for_viewer(serial, intrinsics[serial])
        cam_from_world = _parse_cam_from_world(extrinsics[serial])
        cam_from_robot = cam_from_world @ c2r  # same as visualize_all.py
        world_from_cam = np.linalg.inv(cam_from_robot)
        vis.add_camera(
            name=serial,
            extrinsic=world_from_cam,
            intrinsic=intrinsic_view,
            color=(80, 80, 80),
            size=0.08,
            show_axes=True,
        )


def _resolve_mesh_path(mesh_name: str, mesh_root_dir: str) -> str:
    return os.path.join(mesh_root_dir, mesh_name, f"{mesh_name}.obj")


def _extract_pose_response(res: Dict[str, Any]) -> Dict[str, Any]:
    pose = res.get("object_6d", res)
    out = {
        "pose_world": pose.get("pose_world"),
        "pose_left_cam": pose.get("pose_left_cam"),
        "R_world": pose.get("R_world"),
        "t_world": pose.get("t_world"),
    }
    if out["pose_world"] is None:
        raise RuntimeError(f"invalid response, pose_world missing: {res}")
    return out


def _parse_xyz(raw: str, flag_name: str) -> np.ndarray:
    vals = [x.strip() for x in str(raw).split(",")]
    if len(vals) != 3:
        raise ValueError(f"{flag_name} must be 'x,y,z'")
    return np.array([float(vals[0]), float(vals[1]), float(vals[2])], dtype=np.float32)


def _parse_qpos(raw: str, flag_name: str) -> np.ndarray:
    vals = [x.strip() for x in str(raw).split(",")]
    if not vals:
        raise ValueError(f"{flag_name} must be comma-separated floats")
    return np.array([float(v) for v in vals], dtype=np.float32)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"json object expected: {path}")
    return obj


def _resolve_demo_paths(
    demo_base: str,
    demo_name: str,
    demo_index: int,
    demo_grasp_json: Optional[str],
    demo_object6d_json: Optional[str],
    demo_c2r_path: Optional[str],
) -> Tuple[str, str, str, str]:
    demo_root = os.path.join(demo_base, demo_name, str(demo_index))
    grasp_path = os.path.expanduser(demo_grasp_json) if demo_grasp_json else os.path.join(demo_root, "grasp", "grasp.json")
    object6d_path = (
        os.path.expanduser(demo_object6d_json)
        if demo_object6d_json
        else os.path.join(demo_root, "grasp", "object6d.json")
    )
    if demo_c2r_path:
        c2r_path = os.path.expanduser(demo_c2r_path)
    else:
        c2r_path = os.path.join(demo_base, "C2R.npy")
        if not os.path.exists(c2r_path):
            # Fallback to per-episode C2R when base-level C2R is absent.
            c2r_path = os.path.join(demo_root, "C2R.npy")
    return demo_root, grasp_path, object6d_path, c2r_path


def _extract_pair_payload(grasp_payload: Dict[str, Any]) -> Dict[str, Any]:
    if "pairs" in grasp_payload and isinstance(grasp_payload["pairs"], list) and len(grasp_payload["pairs"]) > 0:
        first = grasp_payload["pairs"][0]
        if isinstance(first, dict) and "pair" in first and isinstance(first["pair"], dict):
            return first["pair"]
        if isinstance(first, dict):
            return first
    if "pair" in grasp_payload and isinstance(grasp_payload["pair"], dict):
        return grasp_payload["pair"]
    return grasp_payload


def _extract_object_pose_world(object6d_payload: Dict[str, Any]) -> np.ndarray:
    for k in ("refined_pose_world", "pose_world", "object_6d"):
        if k == "object_6d":
            v = object6d_payload.get(k)
            if isinstance(v, dict):
                if v.get("pose_world") is not None:
                    return _to_4x4(np.asarray(v["pose_world"], dtype=np.float32))
            continue
        v = object6d_payload.get(k)
        if v is not None:
            return _to_4x4(np.asarray(v, dtype=np.float32))
    raise ValueError("object6d json does not contain refined_pose_world/pose_world")


def _hand_state_to_inspire_f1_qpos(hand_state: np.ndarray) -> np.ndarray:
    # Raw order from dataset: [little, ring, middle, index, thumb2, thumb1].
    hs = np.asarray(hand_state, dtype=np.float32).reshape(-1)
    if hs.shape[0] != 6:
        raise ValueError(f"hand_state length must be 6, got {hs.shape[0]}")
    q = np.zeros(6, dtype=np.float32)
    q[0] = (1800.0 - hs[0]) * np.pi / 1800.0
    q[1] = (1350.0 - hs[1]) * np.pi / 1800.0
    q[2] = (1740.0 - hs[2]) * np.pi / 1800.0
    q[3] = (1740.0 - hs[3]) * np.pi / 1800.0
    q[4] = (1740.0 - hs[4]) * np.pi / 1800.0
    q[5] = (1740.0 - hs[5]) * np.pi / 1800.0
    return q


def _extract_demo_start_qpos(pair: Dict[str, Any], robot_module: RobotModule) -> np.ndarray:
    if pair.get("robot_mesh_qpos") is not None:
        q = np.asarray(pair["robot_mesh_qpos"], dtype=np.float32).reshape(-1)
        if q.shape[0] == robot_module.get_num_joints():
            return q

    arm = np.asarray(pair.get("robot_arm_state"), dtype=np.float32).reshape(-1)
    if arm.shape[0] < 6:
        raise ValueError("robot_arm_state missing/invalid in grasp json")

    q = np.zeros(robot_module.get_num_joints(), dtype=np.float32)
    q[: min(6, q.shape[0])] = arm[: min(6, q.shape[0])]
    hand = pair.get("robot_hand_state")
    if hand is not None and q.shape[0] >= 12:
        hand_q = _hand_state_to_inspire_f1_qpos(np.asarray(hand, dtype=np.float32))
        q[6:12] = hand_q
    return q


def _extract_demo_eef_pose(
    pair: Dict[str, Any],
    robot_module: RobotModule,
    ee_link: str,
    start_qpos: np.ndarray,
    use_action_pose: bool = False,
) -> np.ndarray:
    if use_action_pose and pair.get("robot_arm_action_pose") is not None:
        return _to_4x4(np.asarray(pair["robot_arm_action_pose"], dtype=np.float32))
    robot_module.update_cfg(start_qpos)
    fk_pose = _to_4x4(robot_module.get_transform(ee_link, robot_module.urdf.base_link, collision_geometry=False))
    if (not use_action_pose) and pair.get("robot_arm_action_pose") is not None:
        # Keep both for diagnostics while still using FK as source of truth.
        action_pose = _to_4x4(np.asarray(pair["robot_arm_action_pose"], dtype=np.float32))
        _ = action_pose  # reserved for debugging hooks/logging
    return fk_pose


def _apply_pose_normalization(
    pose_robot_raw: np.ndarray,
    normalize_xyz: bool,
    normalize_xyz_deg: str,
    normalize_xyz_sym180: bool,
    normalize_xyz_sym180_deg: str,
    normalize_y180: bool,
    normalize_y180_deg: float,
    normalize_z: bool,
    normalize_z_deg: float,
) -> np.ndarray:
    if normalize_xyz:
        return _normalize_pose_xyz(pose_robot_raw, _parse_xyz_deg(normalize_xyz_deg))
    if normalize_xyz_sym180:
        return _normalize_pose_xyz_sym180(
            pose_robot_raw, _parse_xyz_deg(normalize_xyz_sym180_deg)
        )
    if normalize_y180:
        return _normalize_pose_y_axis_sym180(
            pose_robot_raw,
            target_yaw_rad=np.deg2rad(float(normalize_y180_deg)),
        )
    if normalize_z:
        return _normalize_pose_z_yaw(pose_robot_raw, target_yaw_rad=np.deg2rad(float(normalize_z_deg)))
    return pose_robot_raw


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


def _normalize_pose_z_yaw(pose_robot: np.ndarray, target_yaw_rad: float = 0.0) -> np.ndarray:
    pose_robot = _to_4x4(pose_robot)
    R = pose_robot[:3, :3]
    t = pose_robot[:3, 3].copy()

    # Yaw in robot frame from x-axis heading projected on XY.
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    delta = float(target_yaw_rad - yaw)
    cz = float(np.cos(delta))
    sz = float(np.sin(delta))
    Rz = np.array(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = Rz @ R
    out[:3, 3] = t  # keep object position fixed
    return out


def _normalize_pose_y_axis_sym180(pose_robot: np.ndarray, target_yaw_rad: float = 0.0) -> np.ndarray:
    pose_robot = _to_4x4(pose_robot)
    R = pose_robot[:3, :3]
    t = pose_robot[:3, 3].copy()

    # Use object's +Y axis heading in robot XY plane.
    # For 180-deg symmetry we choose either yaw or yaw+pi (same facing for symmetric object).
    yaw_y = float(np.arctan2(R[1, 1], R[0, 1]))

    def _wrap_pi(x: float) -> float:
        return float(np.arctan2(np.sin(x), np.cos(x)))

    d0 = _wrap_pi(target_yaw_rad - yaw_y)
    d1 = _wrap_pi(target_yaw_rad - (yaw_y + np.pi))
    delta = d0 if abs(d0) <= abs(d1) else d1

    cz = float(np.cos(delta))
    sz = float(np.sin(delta))
    Rz = np.array(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = Rz @ R
    out[:3, 3] = t
    return out


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


def _parse_xyz_deg(raw: str) -> np.ndarray:
    vals = [x.strip() for x in str(raw).split(",")]
    if len(vals) != 3:
        raise ValueError("--normalize_xyz_deg must be 'rx,ry,rz' in degrees")
    return np.array([float(vals[0]), float(vals[1]), float(vals[2])], dtype=np.float32)


def _normalize_pose_xyz(pose_robot: np.ndarray, target_xyz_deg: np.ndarray) -> np.ndarray:
    pose_robot = _to_4x4(pose_robot)
    t = pose_robot[:3, 3].copy()
    rx, ry, rz = np.deg2rad(np.asarray(target_xyz_deg, dtype=np.float32))
    # Fixed target orientation in robot frame (XYZ intrinsic order equivalent: Rz @ Ry @ Rx).
    R_target = _rot_z(float(rz)) @ _rot_y(float(ry)) @ _rot_x(float(rx))
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = R_target
    out[:3, 3] = t
    return out


def _normalize_pose_xyz_sym180(pose_robot: np.ndarray, target_xyz_deg: np.ndarray) -> np.ndarray:
    pose_robot = _to_4x4(pose_robot)
    R = pose_robot[:3, :3]
    t = pose_robot[:3, 3].copy()
    rx, ry, rz = np.deg2rad(np.asarray(target_xyz_deg, dtype=np.float32))
    # Reference axis convention only; do not hard-align exact orientation.
    R_ref = _rot_z(float(rz)) @ _rot_y(float(ry)) @ _rot_x(float(rx))

    # D2 symmetry of rectangular cuboid: 180-deg flips around local x/y/z.
    sym = [
        np.diag([1.0, 1.0, 1.0]).astype(np.float32),
        np.diag([1.0, -1.0, -1.0]).astype(np.float32),
        np.diag([-1.0, 1.0, -1.0]).astype(np.float32),
        np.diag([-1.0, -1.0, 1.0]).astype(np.float32),
    ]

    best_R = R
    best_score = -1e30
    for S in sym:
        Rc = R @ S
        # Higher trace => closer axis-direction convention to reference.
        score = float(np.trace(R_ref.T @ Rc))
        if score > best_score:
            best_score = score
            best_R = Rc

    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = best_R
    out[:3, 3] = t
    return out


def _build_minimal_curobo_robot_cfg(robot_urdf_path: str, ee_link: str) -> Dict[str, Any]:
    robot_urdf_path = os.path.abspath(os.path.expanduser(robot_urdf_path))
    robot_module = RobotModule(robot_urdf_path)
    jn = list(robot_module.get_joint_names())
    if ee_link not in robot_module.urdf.link_map:
        raise ValueError(f"ee_link '{ee_link}' not found in URDF links")
    urdf_dir = str(Path(robot_urdf_path).parent)
    urdf_file = str(Path(robot_urdf_path).name)
    return {
        "robot_cfg": {
            "kinematics": {
                "use_usd_kinematics": False,
                # cuRobo resolves urdf_path from asset roots; use external_asset_path for local files.
                "urdf_path": urdf_file,
                "asset_root_path": "",
                "external_asset_path": urdf_dir,
                "base_link": robot_module.urdf.base_link,
                "ee_link": ee_link,
                "collision_link_names": None,
                "collision_spheres": None,
                "collision_sphere_buffer": 0.005,
                "extra_collision_spheres": {},
                "self_collision_ignore": {},
                "self_collision_buffer": {},
                "use_global_cumul": True,
                "mesh_link_names": None,
                "lock_joints": None,
                "extra_links": None,
                "cspace": {
                    "joint_names": jn,
                    "retract_config": [0.0] * len(jn),
                    "null_space_weight": [1.0] * len(jn),
                    "cspace_distance_weight": [1.0] * len(jn),
                    "max_acceleration": 15.0,
                    "max_jerk": 500.0,
                },
            }
        }
    }


def _normalize_curobo_robot_cfg_paths(robot_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "robot_cfg" in robot_cfg:
        cfg = robot_cfg
    else:
        cfg = {"robot_cfg": robot_cfg}
    kin = cfg["robot_cfg"].setdefault("kinematics", {})

    urdf_path = kin.get("urdf_path")
    ext_asset = kin.get("external_asset_path")
    asset_root = kin.get("asset_root_path")

    if urdf_path:
        up = os.path.expanduser(str(urdf_path))
        # If absolute path is provided, split into external_asset_path + basename.
        if os.path.isabs(up):
            kin["external_asset_path"] = str(Path(up).parent)
            kin["asset_root_path"] = ""
            kin["urdf_path"] = str(Path(up).name)
        else:
            # If relative path exists in cwd, promote to external asset path.
            abs_guess = os.path.abspath(up)
            if os.path.exists(abs_guess):
                kin["external_asset_path"] = str(Path(abs_guess).parent)
                kin["asset_root_path"] = ""
                kin["urdf_path"] = str(Path(abs_guess).name)
            elif ext_asset:
                eap = os.path.expanduser(str(ext_asset))
                joined = os.path.join(eap, up)
                if os.path.exists(joined):
                    kin["external_asset_path"] = os.path.abspath(eap)
                    kin["asset_root_path"] = ""
                    kin["urdf_path"] = str(Path(joined).name)
            elif asset_root:
                arp = os.path.expanduser(str(asset_root))
                joined = os.path.join(arp, up)
                if os.path.exists(joined):
                    kin["external_asset_path"] = os.path.abspath(arp)
                    kin["asset_root_path"] = ""
                    kin["urdf_path"] = str(Path(joined).name)
    return cfg


def _normalize_curobo_robot_cfg_paths_with_cfg_file(robot_cfg: Dict[str, Any], cfg_file: str) -> Dict[str, Any]:
    cfg = _normalize_curobo_robot_cfg_paths(robot_cfg)
    kin = cfg["robot_cfg"]["kinematics"]
    coll_spheres = kin.get("collision_spheres")
    if not (isinstance(coll_spheres, str) and len(coll_spheres) > 0):
        return cfg
    cs_raw = os.path.expanduser(coll_spheres)
    if os.path.isabs(cs_raw) and os.path.exists(cs_raw):
        return cfg

    cfg_dir = os.path.abspath(os.path.dirname(cfg_file))
    base_dirs = [cfg_dir]
    p = Path(cfg_dir)
    for i in range(1, min(8, len(p.parents) + 1)):
        base_dirs.append(str(p.parents[i - 1]))
    base_dirs.append(str(PROJECT_ROOT))

    candidates: List[str] = []

    def _add(path: str) -> None:
        ap = os.path.abspath(os.path.expanduser(path))
        if ap not in candidates:
            candidates.append(ap)

    _add(os.path.join(cfg_dir, cs_raw))
    _add(cs_raw)
    for b in base_dirs:
        _add(os.path.join(b, cs_raw))
        _add(os.path.join(b, "configs", "robot", cs_raw))
        _add(os.path.join(b, "content", "configs", "robot", cs_raw))
        _add(os.path.join(b, "assets", "content", "configs", "robot", cs_raw))
    for c in candidates:
        if os.path.exists(c):
            kin["collision_spheres"] = c
            break
    return cfg


def _resolve_start_qpos_for_cspace(start_qpos_in: np.ndarray, robot_cfg: Dict[str, Any]) -> np.ndarray:
    kin = robot_cfg["robot_cfg"]["kinematics"]
    cspace = kin.get("cspace", {})
    joint_names = list(cspace.get("joint_names", []))
    dof = len(joint_names)
    retract = cspace.get("retract_config")
    sq = np.asarray(start_qpos_in, dtype=np.float32).reshape(-1)
    if dof == 0 or sq.shape[0] == dof:
        return sq
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


def _expected_active_dof_from_robot_cfg(robot_cfg: Dict[str, Any]) -> Optional[int]:
    kin = robot_cfg.get("robot_cfg", {}).get("kinematics", {})
    cspace = kin.get("cspace", {})
    joint_names = list(cspace.get("joint_names", []))
    if len(joint_names) == 0:
        return None
    lock = kin.get("lock_joints") or {}
    lock_names = set(str(k) for k in lock.keys())
    active = [jn for jn in joint_names if str(jn) not in lock_names]
    if len(active) == 0:
        return None
    return int(len(active))


def _run_curobo_plan(
    robot_cfg: Dict[str, Any],
    world_cfg_dict: Dict[str, Any],
    start_qpos: np.ndarray,
    target_eef_pose_robot: np.ndarray,
    device: str = "cuda:0",
    require_dof: Optional[int] = None,
    seeds: int = 256,
    self_collision: bool = True,
) -> Dict[str, Any]:
    try:
        local_curobo_src = str(PROJECT_ROOT / "curobo" / "src")
        if local_curobo_src not in sys.path:
            sys.path.insert(0, local_curobo_src)

        import torch
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import WorldConfig
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose
        from curobo.types.state import JointState
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    except Exception as e:
        return {"success": False, "error": f"curobo import failed: {e}"}

    try:
        dev = torch.device(device if ("cuda" in str(device) and torch.cuda.is_available()) else "cpu")
        tensor_args = TensorDeviceType(device=dev)
        world_cfg = WorldConfig.from_dict(world_cfg_dict)

        mg_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=False,
            interpolation_dt=0.01,
            num_ik_seeds=int(seeds),
            num_trajopt_seeds=64,
            num_graph_seeds=32,
            position_threshold=0.01,
            rotation_threshold=0.1,
            self_collision_check=bool(self_collision),
            self_collision_opt=bool(self_collision),
            store_ik_debug=True,
            store_debug_in_result=True,
        )
        motion_gen = MotionGen(mg_cfg)

        active_joint_names = list(getattr(motion_gen.kinematics, "joint_names", []))
        active_dof = len(active_joint_names) if active_joint_names else int(np.asarray(start_qpos).reshape(-1).shape[0])
        if require_dof is not None and active_dof != int(require_dof):
            return {
                "success": False,
                "error": f"curobo active dof mismatch (active={active_dof}, required={int(require_dof)})",
                "active_joint_names": active_joint_names,
                "active_dof": active_dof,
            }
        start_np = np.asarray(start_qpos, dtype=np.float32).reshape(-1)
        if start_np.shape[0] != active_dof:
            start_np = _select_active_qpos_from_cspace(start_np, robot_cfg, active_joint_names)
            if start_np.shape[0] != active_dof:
                return {
                    "success": False,
                    "error": f"start_qpos dof mismatch (start={start_np.shape[0]}, active={active_dof})",
                    "active_joint_names": active_joint_names,
                    "active_dof": active_dof,
                }

        start = JointState.from_position(
            torch.tensor(start_np, device=tensor_args.device, dtype=torch.float32),
            joint_names=active_joint_names if len(active_joint_names) > 0 else None,
        ).unsqueeze(0)
        goal = Pose(
            position=torch.tensor(
                target_eef_pose_robot[:3, 3], device=tensor_args.device, dtype=torch.float32
            ).view(1, 3),
            quaternion=torch.tensor(
                _rotmat_to_wxyz(target_eef_pose_robot[:3, :3]),
                device=tensor_args.device,
                dtype=torch.float32,
            ).view(1, 4),
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
        out = {"success": success}
        out["active_joint_names"] = active_joint_names
        out["status"] = str(result.status)
        out["valid_query"] = bool(getattr(result, "valid_query", True))
        out["attempts"] = int(getattr(result, "attempts", 0))
        out["timing_s"] = {
            "total": float(getattr(result, "total_time", 0.0)),
            "ik": float(getattr(result, "ik_time", 0.0)),
            "graph": float(getattr(result, "graph_time", 0.0)),
            "trajopt": float(getattr(result, "trajopt_time", 0.0)),
            "finetune": float(getattr(result, "finetune_time", 0.0)),
        }

        # Log final trans/rot error when available from MotionGenResult.
        try:
            pe_t = getattr(result, "position_error", None)
            if pe_t is not None:
                pe = pe_t.detach().cpu().reshape(-1).numpy().astype(np.float64)
                out["position_error_min"] = float(np.min(pe))
                out["position_error_mean"] = float(np.mean(pe))
                out["position_error_max"] = float(np.max(pe))
                out["position_error_best"] = float(np.min(pe))
        except Exception:
            pass
        try:
            re_t = getattr(result, "rotation_error", None)
            if re_t is not None:
                re = re_t.detach().cpu().reshape(-1).numpy().astype(np.float64)
                out["rotation_error_min"] = float(np.min(re))
                out["rotation_error_mean"] = float(np.mean(re))
                out["rotation_error_max"] = float(np.max(re))
                out["rotation_error_best"] = float(np.min(re))
        except Exception:
            pass

        # Also dump IK candidate error stats whenever debug exists (success/failure both).
        ik_diag = _extract_curobo_ik_debug(result)
        if ik_diag is not None:
            out["ik_debug"] = ik_diag
            out["best_qpos"] = ik_diag.get("best_qpos")
            out["best_position_error"] = ik_diag.get("best_position_error")
            out["best_rotation_error"] = ik_diag.get("best_rotation_error")
            out["ik_success_count"] = ik_diag.get("success_count")
            out["ik_position_error_min"] = ik_diag.get("position_error_min")
            out["ik_position_error_mean"] = ik_diag.get("position_error_mean")
            out["ik_rotation_error_min"] = ik_diag.get("rotation_error_min")
            out["ik_rotation_error_mean"] = ik_diag.get("rotation_error_mean")

        if success:
            interp = result.get_interpolated_plan()
            out["joint_names"] = list(interp.joint_names)
            out["active_dof"] = int(interp.position.shape[-1])
            out["trajectory_active"] = interp.position.detach().cpu().numpy().tolist()
            out["trajectory"] = interp.position.detach().cpu().numpy().tolist()
        return out
    except Exception as e:
        return {"success": False, "error": f"curobo planning failed: {e}"}


def _extract_curobo_ik_debug(result: Any) -> Optional[Dict[str, Any]]:
    try:
        dbg = getattr(result, "debug_info", None)
        if not isinstance(dbg, dict):
            return None
        ik_result = dbg.get("ik_result")
        if ik_result is None:
            return None
        import torch

        def _to_np(x: Any) -> Optional[np.ndarray]:
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        sol = _to_np(getattr(ik_result, "solution", None))
        pos_err = _to_np(getattr(ik_result, "position_error", None))
        rot_err = _to_np(getattr(ik_result, "rotation_error", None))
        success = _to_np(getattr(ik_result, "success", None))
        if sol is None or pos_err is None or rot_err is None:
            return None

        dof = int(sol.shape[-1])
        sol2 = sol.reshape(-1, dof)
        pe = pos_err.reshape(-1).astype(np.float64)
        re = rot_err.reshape(-1).astype(np.float64)
        score = pe + re
        best_idx = int(np.argmin(score))

        out = {
            "num_candidates": int(sol2.shape[0]),
            "dof": dof,
            "best_index": best_idx,
            "best_qpos": sol2[best_idx].astype(np.float32).tolist(),
            "best_position_error": float(pe[best_idx]),
            "best_rotation_error": float(re[best_idx]),
            "best_score": float(score[best_idx]),
            "position_error_min": float(np.min(pe)),
            "position_error_mean": float(np.mean(pe)),
            "rotation_error_min": float(np.min(re)),
            "rotation_error_mean": float(np.mean(re)),
        }
        if success is not None:
            succ = success.reshape(-1).astype(np.bool_)
            out["success_count"] = int(np.sum(succ))
        return out
    except Exception:
        return None


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
        raise ValueError(
            f"active qpos/name mismatch: q={q.shape[0]}, active_names={len(active_joint_names)}"
        )
    idx_map = {str(jn): i for i, jn in enumerate(full_joint_names)}
    for i, jn in enumerate(active_joint_names):
        idx = idx_map.get(str(jn))
        if idx is None:
            raise ValueError(f"active joint '{jn}' not found in full joint names")
        out[idx] = float(q[i])
    return out


def _make_thumb1_only_hand_start_qpos(
    start_qpos: np.ndarray,
    demo_qpos: np.ndarray,
    full_joint_names: List[str],
    arm_dof: int = 6,
) -> np.ndarray:
    out = np.asarray(start_qpos, dtype=np.float32).copy()
    demo = np.asarray(demo_qpos, dtype=np.float32).reshape(-1)
    n = min(out.shape[0], demo.shape[0])
    if n <= arm_dof:
        return out
    out[arm_dof:n] = 0.0
    thumb1_idx = [
        i
        for i, jn in enumerate(full_joint_names[:n])
        if i >= arm_dof and ("thumb_1" in str(jn) or "thumb1" in str(jn))
    ]
    if len(thumb1_idx) == 0 and arm_dof < n:
        thumb1_idx = [arm_dof]
    for idx in thumb1_idx:
        out[idx] = float(demo[idx])
    return out


def _build_full_traj_from_curobo_result(
    curobo_result: Dict[str, Any],
    plan_start_qpos: np.ndarray,
    full_joint_names: List[str],
    arm_only: bool,
) -> Optional[np.ndarray]:
    if not (isinstance(curobo_result, dict) and curobo_result.get("success") and curobo_result.get("trajectory") is not None):
        return None
    traj = np.asarray(curobo_result["trajectory"], dtype=np.float32)
    if traj.ndim != 2 or traj.shape[0] == 0:
        return None
    if traj.shape[-1] == plan_start_qpos.shape[0]:
        return traj
    if arm_only and traj.shape[-1] == 6:
        return np.stack([_expand_arm_qpos_to_full(q, plan_start_qpos) for q in traj], axis=0)
    active_joint_names = list(curobo_result.get("active_joint_names", []) or [])
    return np.stack(
        [
            _expand_active_qpos_to_full(q, plan_start_qpos, full_joint_names, active_joint_names)
            for q in traj
        ],
        axis=0,
    )


def _append_hand_post_trajectory(
    full_traj: np.ndarray,
    hand_target_full_qpos: np.ndarray,
    hand_start_idx: int = 6,
    interp_steps: int = 90,
) -> np.ndarray:
    traj = np.asarray(full_traj, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[0] == 0 or int(interp_steps) <= 0:
        return traj
    target = np.asarray(hand_target_full_qpos, dtype=np.float32).reshape(-1)
    n = min(traj.shape[1], target.shape[0])
    if n <= hand_start_idx:
        return traj
    hand_idx = list(range(hand_start_idx, n))
    q0 = traj[-1].copy()
    q1 = q0.copy()
    q1[hand_idx] = target[hand_idx]
    if np.allclose(q0[hand_idx], q1[hand_idx], atol=1e-6):
        return traj
    alpha = np.linspace(0.0, 1.0, int(interp_steps) + 1, dtype=np.float32)[1:]
    tail = np.repeat(q0[None, :], repeats=alpha.shape[0], axis=0)
    tail[:, hand_idx] = (1.0 - alpha[:, None]) * q0[None, hand_idx] + alpha[:, None] * q1[None, hand_idx]
    return np.concatenate([traj, tail], axis=0)


def _make_arm_only_robot_cfg(
    robot_cfg: Dict[str, Any], start_qpos: np.ndarray, joint_names: List[str], arm_dof: int = 6
) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(robot_cfg))  # simple deep copy
    kin = cfg["robot_cfg"]["kinematics"]
    arm_names = list(joint_names[:arm_dof])
    hand_names = list(joint_names[arm_dof:])

    cspace = kin.setdefault("cspace", {})
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default="object_6d/debug", help="relative path under shared_data")
    parser.add_argument("--name", "--mesh_name", dest="mesh_name", required=True)
    parser.add_argument("--rpc_addr", default="tcp://192.168.0.14:5570")
    parser.add_argument("--rpc_timeout_ms", type=int, default=300000)
    parser.add_argument("--rcc_entry", default="image_main.py")
    parser.add_argument("--mesh_root_dir", default=os.path.join(shared_dir, "mesh_blender"))
    parser.add_argument("--robot_urdf", default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--eef_link", default="link6")
    parser.add_argument("--demo_base", default=DEFAULT_DEMO_BASE)
    parser.add_argument("--demo_name", default=None)
    parser.add_argument("--demo_index", type=int, default=0)
    parser.add_argument("--demo_grasp_json", default=None)
    parser.add_argument("--demo_object6d_json", default=None)
    parser.add_argument("--demo_c2r_path", default=None)
    parser.add_argument(
        "--demo_use_action_pose",
        action="store_true",
        help="use robot_arm_action_pose from grasp.json instead of FK(robot_arm_state)",
    )
    parser.add_argument("--start_qpos", default=None, help="optional csv qpos override")
    parser.add_argument("--run_curobo", action="store_true")
    parser.add_argument("--arm_only", action="store_true", help="plan arm only (6 dof)")
    parser.add_argument("--curobo_robot_cfg", default=DEFAULT_CUROBO_ROBOT_CFG, help="curobo robot cfg yaml")
    parser.add_argument("--curobo_device", default="cuda:0")
    parser.add_argument("--curobo_seeds", type=int, default=256)
    parser.add_argument("--curobo_disable_self_collision", action="store_true")
    parser.add_argument("--curobo_ignore_world", action="store_true", help="debug: disable world collisions in curobo")
    parser.add_argument("--lock_joints", action="store_true", help="lock hand joints to start_qpos")
    parser.add_argument(
        "--hand_qpos_config_cost",
        action="store_true",
        help="apply hand-only null-space config cost around target hand qpos (pose target is kept)",
    )
    parser.add_argument("--hand_qpos_weight", type=float, default=1.0)
    parser.add_argument(
        "--hand_qpos_target",
        default=None,
        help="optional csv full-dof qpos target used as retract config (default: start_qpos)",
    )
    parser.add_argument(
        "--hand_post_interp_steps",
        type=int,
        default=90,
        help="append this many hand-only interpolation steps after curobo trajectory",
    )
    parser.add_argument("--table_size_xyz", default="2.4,1.6,0.1")
    parser.add_argument("--table_pos_xyz", default="0.0,0.0,-0.05")
    parser.add_argument("--c2r_path", default=None)
    parser.add_argument("--normalize_z", action="store_true")
    parser.add_argument("--normalize_z_deg", type=float, default=0.0)
    parser.add_argument(
        "--normalize_y180",
        action="store_true",
        help="normalize yaw using object +Y axis with 180-degree symmetry",
    )
    parser.add_argument(
        "--normalize_y180_deg",
        type=float,
        default=0.0,
        help="target yaw(deg) for --normalize_y180 in robot frame",
    )
    parser.add_argument("--normalize_xyz", action="store_true")
    parser.add_argument("--normalize_xyz_deg", default="0,0,0")
    parser.add_argument(
        "--normalize_xyz_sym180",
        action="store_true",
        help="canonicalize cuboid orientation with xyz 180-degree symmetry (axis convention only)",
    )
    parser.add_argument(
        "--normalize_xyz_sym180_deg",
        default="0,0,0",
        help="reference xyz(deg) for --normalize_xyz_sym180",
    )
    parser.add_argument(
        "--mujoco_after_vis",
        action="store_true",
        help="after closing Viser, open MuJoCo and replay the trajectory with world object/table",
    )
    parser.add_argument(
        "--mujoco_robot_mjcf",
        default=str(PROJECT_ROOT / "rsc" / "curobo" / "content" / "assets" / "robot" / "inspire_description" / "xarm_inspire_f1.mjcf"),
        help="MuJoCo robot mjcf path used for post-Viser replay",
    )
    parser.add_argument("--mujoco_waypoint_hold_steps", type=int, default=4)
    parser.add_argument("--mujoco_object_mass", type=float, default=0.05)
    parser.add_argument(
        "--mujoco_dynamic_step",
        action="store_true",
        help="use dynamic servo replay with MuJoCo actuators for physical lift validation",
    )
    parser.add_argument("--mujoco_validate_lift", action="store_true")
    parser.add_argument("--mujoco_lift_success_z", type=float, default=0.05)
    parser.add_argument("--mujoco_start_settle_steps", type=int, default=180)
    parser.add_argument(
        "--mujoco_hand_kinematic_in_dynamic",
        action="store_true",
        help="force hand joints to kinematic tracking in dynamic replay (default: dynamic hand actuators)",
    )
    parser.add_argument(
        "--mujoco_no_hand_position_actuator",
        action="store_true",
        help="legacy alias; hand default is general actuator",
    )
    parser.add_argument(
        "--mujoco_general_hand_actuator",
        action="store_true",
        help="use general (PD-style) actuator for hand joints (default behavior)",
    )
    parser.add_argument(
        "--mujoco_hand_position_actuator",
        action="store_true",
        help="use position actuator for hand joints instead of default general actuator",
    )
    parser.add_argument("--mujoco_no_xarm7_profile", action="store_true")
    parser.add_argument("--mujoco_arm_kp", type=float, default=1200.0)
    parser.add_argument("--mujoco_arm_kd", type=float, default=180.0)
    parser.add_argument("--mujoco_arm_force", type=float, default=80.0)
    parser.add_argument("--mujoco_hand_kp", type=float, default=4.0)
    parser.add_argument("--mujoco_hand_kd", type=float, default=10.0)
    parser.add_argument("--mujoco_hand_force", type=float, default=15.0)
    parser.add_argument("--mujoco_hand_ctrl_max_step", type=float, default=0.008)
    parser.add_argument("--mujoco_hand_position_kp", type=float, default=8.0)
    parser.add_argument("--mujoco_decimation", type=int, default=3)
    parser.add_argument("--no_vis", action="store_true")
    args = parser.parse_args()

    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rel_save_path = os.path.join(args.save_path, date_str)
    abs_save_path = os.path.join(shared_dir, rel_save_path)

    rcc = remote_camera_controller(args.rcc_entry)
    save_current_camparam(abs_save_path)
    print(f"Capturing image to {rel_save_path}")

    try:
        rcc.start("image", False, f"shared_data/{rel_save_path}/raw")
        rcc.stop()
    finally:
        rcc.end()

    image_dir = abs_save_path
    print(f"image_dir: {image_dir}")

    req = {
        "command": "infer",
        "image_path": _to_shared_data_path(image_dir),
        "mesh_name": args.mesh_name,
    }
    res = _send_rpc_once(args.rpc_addr, req, timeout_ms=args.rpc_timeout_ms)
    pose_out = _extract_pose_response(res)

    object_json_path = os.path.join(abs_save_path, "object_6d.json")
    with open(object_json_path, "w", encoding="utf-8") as f:
        json.dump(pose_out, f, ensure_ascii=False, indent=2)
    print(f"saved: {object_json_path}")

    if args.c2r_path is not None:
        c2r = np.load(os.path.expanduser(args.c2r_path))
    else:
        c2r = load_current_C2R()
    c2r = _to_4x4(c2r)
    np.save(os.path.join(abs_save_path, "C2R.npy"), c2r)

    pose_world = _to_4x4(np.asarray(pose_out["pose_world"], dtype=np.float32))
    r2c = np.linalg.inv(c2r)
    pose_robot_raw = r2c @ pose_world  # same as visualize_all.py
    pose_robot = _apply_pose_normalization(
        pose_robot_raw,
        normalize_xyz=bool(args.normalize_xyz),
        normalize_xyz_deg=args.normalize_xyz_deg,
        normalize_xyz_sym180=bool(args.normalize_xyz_sym180),
        normalize_xyz_sym180_deg=args.normalize_xyz_sym180_deg,
        normalize_y180=bool(args.normalize_y180),
        normalize_y180_deg=float(args.normalize_y180_deg),
        normalize_z=bool(args.normalize_z),
        normalize_z_deg=float(args.normalize_z_deg),
    )

    mesh_path = _resolve_mesh_path(args.mesh_name, os.path.expanduser(args.mesh_root_dir))
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"mesh not found: {mesh_path}")
    robot_urdf_path = os.path.expanduser(args.robot_urdf)
    if not os.path.exists(robot_urdf_path):
        raise FileNotFoundError(f"robot urdf not found: {robot_urdf_path}")

    print(json.dumps(pose_out, ensure_ascii=False, indent=2))

    robot_pose_json_path = os.path.join(abs_save_path, "object_6d_robot.json")
    robot_out = {
        "pose_robot": pose_robot.tolist(),
        "R_robot": pose_robot[:3, :3].tolist(),
        "t_robot": pose_robot[:3, 3].tolist(),
        "pose_robot_raw": pose_robot_raw.tolist(),
        "normalize_xyz": bool(args.normalize_xyz),
        "normalize_xyz_deg": args.normalize_xyz_deg,
        "normalize_xyz_sym180": bool(args.normalize_xyz_sym180),
        "normalize_xyz_sym180_deg": args.normalize_xyz_sym180_deg,
        "normalize_y180": bool(args.normalize_y180),
        "normalize_y180_deg": float(args.normalize_y180_deg),
        "normalize_z": bool(args.normalize_z),
        "normalize_z_deg": float(args.normalize_z_deg),
    }
    with open(robot_pose_json_path, "w", encoding="utf-8") as f:
        json.dump(robot_out, f, ensure_ascii=False, indent=2)
    print(f"saved: {robot_pose_json_path}")

    demo_name = args.demo_name if args.demo_name is not None else args.mesh_name
    demo_root, demo_grasp_json_path, demo_object6d_json_path, demo_c2r_path = _resolve_demo_paths(
        demo_base=os.path.expanduser(args.demo_base),
        demo_name=demo_name,
        demo_index=int(args.demo_index),
        demo_grasp_json=args.demo_grasp_json,
        demo_object6d_json=args.demo_object6d_json,
        demo_c2r_path=args.demo_c2r_path,
    )
    if not os.path.exists(demo_grasp_json_path):
        raise FileNotFoundError(f"demo grasp json not found: {demo_grasp_json_path}")
    if not os.path.exists(demo_object6d_json_path):
        raise FileNotFoundError(f"demo object6d json not found: {demo_object6d_json_path}")
    if not os.path.exists(demo_c2r_path):
        raise FileNotFoundError(f"demo C2R not found: {demo_c2r_path}")

    demo_grasp_payload = _load_json(demo_grasp_json_path)
    demo_pair = _extract_pair_payload(demo_grasp_payload)
    demo_object_payload = _load_json(demo_object6d_json_path)

    demo_obj_world = _extract_object_pose_world(demo_object_payload)
    demo_c2r = _to_4x4(np.load(demo_c2r_path))
    demo_obj_robot_raw = np.linalg.inv(demo_c2r) @ demo_obj_world
    demo_obj_robot = _apply_pose_normalization(
        demo_obj_robot_raw,
        normalize_xyz=bool(args.normalize_xyz),
        normalize_xyz_deg=args.normalize_xyz_deg,
        normalize_xyz_sym180=bool(args.normalize_xyz_sym180),
        normalize_xyz_sym180_deg=args.normalize_xyz_sym180_deg,
        normalize_y180=bool(args.normalize_y180),
        normalize_y180_deg=float(args.normalize_y180_deg),
        normalize_z=bool(args.normalize_z),
        normalize_z_deg=float(args.normalize_z_deg),
    )

    robot_module = RobotModule(robot_urdf_path)
    demo_start_qpos = _extract_demo_start_qpos(demo_pair, robot_module)
    demo_eef_robot = _extract_demo_eef_pose(
        demo_pair,
        robot_module,
        args.eef_link,
        demo_start_qpos,
        use_action_pose=bool(args.demo_use_action_pose),
    )
    obj_to_eef = np.linalg.inv(demo_obj_robot) @ demo_eef_robot
    target_eef_pose_robot = pose_robot @ obj_to_eef
    # First curobo target offset: move 2cm in robot -X direction.
    target_eef_pose_robot[0, 3] -= 0.005

    if args.start_qpos is not None:
        plan_start_qpos = _parse_qpos(args.start_qpos, "--start_qpos")
    else:
        # Default initial state for planning:
        # fixed arm qpos + thumb_1 from demo hand qpos (other hand joints are set to 0).
        plan_start_qpos = np.zeros(robot_module.get_num_joints(), dtype=np.float32)
        n_arm = min(6, plan_start_qpos.shape[0])
        plan_start_qpos[:n_arm] = DEFAULT_INIT_ARM_QPOS[:n_arm]
        plan_start_qpos = _make_thumb1_only_hand_start_qpos(
            start_qpos=plan_start_qpos,
            demo_qpos=demo_start_qpos,
            full_joint_names=list(robot_module.get_joint_names()),
            arm_dof=6,
        )
    if plan_start_qpos.shape[0] != robot_module.get_num_joints():
        raise ValueError(
            f"start_qpos length {plan_start_qpos.shape[0]} does not match robot joints {robot_module.get_num_joints()}"
        )

    transfer_json_path = os.path.join(abs_save_path, "grasp_transfer_target.json")
    transfer_out = {
        "demo_root": demo_root,
        "demo_grasp_json": demo_grasp_json_path,
        "demo_object6d_json": demo_object6d_json_path,
        "demo_c2r_path": demo_c2r_path,
        "demo_object_pose_robot": demo_obj_robot.tolist(),
        "demo_eef_pose_robot": demo_eef_robot.tolist(),
        "demo_eef_source": "robot_arm_action_pose" if bool(args.demo_use_action_pose) else "fk_from_robot_state",
        "obj_to_eef": obj_to_eef.tolist(),
        "current_object_pose_robot": pose_robot.tolist(),
        "target_eef_pose_robot": target_eef_pose_robot.tolist(),
        "start_qpos": plan_start_qpos.tolist(),
    }
    with open(transfer_json_path, "w", encoding="utf-8") as f:
        json.dump(transfer_out, f, ensure_ascii=False, indent=2)
    print(f"saved: {transfer_json_path}")
    print(
        "planning start arm qpos:",
        np.asarray(plan_start_qpos[:6], dtype=np.float32).round(6).tolist(),
        "| demo eef source:",
        transfer_out["demo_eef_source"],
    )

    table_size_xyz = _parse_xyz(args.table_size_xyz, "--table_size_xyz")
    table_pos_xyz = _parse_xyz(args.table_pos_xyz, "--table_pos_xyz")
    obj_wxyz = _rotmat_to_wxyz(pose_robot[:3, :3])
    planning_scene = {
        "robot": {
            "urdf_path": robot_urdf_path,
            "base_frame": "base_link",
        },
        "world": {
            "cuboid": {
                "table": {
                    "dims": table_size_xyz.tolist(),
                    "pose": [
                        float(table_pos_xyz[0]),
                        float(table_pos_xyz[1]),
                        float(table_pos_xyz[2]),
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                }
            },
            "mesh": {
                "object": {
                    "file_path": mesh_path,
                    "pose": [
                        float(pose_robot[0, 3]),
                        float(pose_robot[1, 3]),
                        float(pose_robot[2, 3]),
                        float(obj_wxyz[0]),
                        float(obj_wxyz[1]),
                        float(obj_wxyz[2]),
                        float(obj_wxyz[3]),
                    ],
                }
            },
        },
        "target": {
            "mesh_name": args.mesh_name,
            "mesh_path": mesh_path,
            "pose_robot": pose_robot.tolist(),
            "eef_link": args.eef_link,
            "target_eef_pose_robot": target_eef_pose_robot.tolist(),
            "start_qpos": plan_start_qpos.tolist(),
        },
    }
    planning_json_path = os.path.join(abs_save_path, "planning_scene_curobo.json")
    with open(planning_json_path, "w", encoding="utf-8") as f:
        json.dump(planning_scene, f, ensure_ascii=False, indent=2)
    print(f"saved: {planning_json_path}")

    curobo_result = None
    if args.run_curobo:
        from paradex.utils.file_io import load_yaml

        if args.curobo_robot_cfg is not None and os.path.exists(os.path.expanduser(args.curobo_robot_cfg)):
            cfg_path = os.path.expanduser(args.curobo_robot_cfg)
            robot_cfg = load_yaml(cfg_path)
            robot_cfg = _normalize_curobo_robot_cfg_paths_with_cfg_file(robot_cfg, cfg_path)
        else:
            robot_cfg = _build_minimal_curobo_robot_cfg(robot_urdf_path, args.eef_link)
        robot_cfg = _normalize_curobo_robot_cfg_paths(robot_cfg)
        robot_cfg["robot_cfg"]["kinematics"]["ee_link"] = str(args.eef_link)

        plan_start_qpos = _resolve_start_qpos_for_cspace(plan_start_qpos, robot_cfg)
        if args.arm_only:
            robot_cfg = _make_arm_only_robot_cfg(
                robot_cfg=robot_cfg,
                start_qpos=plan_start_qpos,
                joint_names=list(robot_module.get_joint_names()),
                arm_dof=6,
            )
            plan_start_qpos = _resolve_start_qpos_for_cspace(plan_start_qpos, robot_cfg)
        if args.lock_joints and not args.arm_only:
            robot_cfg = _lock_hand_joints(robot_cfg, plan_start_qpos, arm_dof=6)
        if args.hand_qpos_config_cost and not args.arm_only:
            hand_target = plan_start_qpos
            if args.hand_qpos_target is not None:
                hand_target = _parse_qpos(args.hand_qpos_target, "--hand_qpos_target")
                hand_target = _resolve_start_qpos_for_cspace(hand_target, robot_cfg)
            robot_cfg = _apply_hand_qpos_config_cost(
                robot_cfg=robot_cfg,
                target_qpos=hand_target,
                hand_weight=float(args.hand_qpos_weight),
                arm_dof=6,
            )

        curobo_start_qpos = plan_start_qpos[:6] if args.arm_only else plan_start_qpos
        expected_active_dof = _expected_active_dof_from_robot_cfg(robot_cfg)
        curobo_world = {"cuboid": {}, "mesh": {}} if args.curobo_ignore_world else planning_scene["world"]
        curobo_result = _run_curobo_plan(
            robot_cfg=robot_cfg,
            world_cfg_dict=curobo_world,
            start_qpos=curobo_start_qpos,
            target_eef_pose_robot=target_eef_pose_robot,
            device=args.curobo_device,
            require_dof=expected_active_dof,
            seeds=int(args.curobo_seeds),
            self_collision=(not bool(args.curobo_disable_self_collision)),
        )
        curobo_result["world_collision_enabled"] = not bool(args.curobo_ignore_world)
        curobo_result["self_collision_enabled"] = not bool(args.curobo_disable_self_collision)
        curobo_result["lock_joints_enabled"] = bool(args.lock_joints and not args.arm_only)
        curobo_result["hand_qpos_config_cost_enabled"] = bool(args.hand_qpos_config_cost and not args.arm_only)
        full_traj = _build_full_traj_from_curobo_result(
            curobo_result=curobo_result,
            plan_start_qpos=plan_start_qpos,
            full_joint_names=list(robot_module.get_joint_names()),
            arm_only=bool(args.arm_only),
        )
        if full_traj is not None:
            full_traj = _append_hand_post_trajectory(
                full_traj=full_traj,
                hand_target_full_qpos=demo_start_qpos,
                hand_start_idx=6,
                interp_steps=int(args.hand_post_interp_steps),
            )
            # After hand-close interpolation, run an additional lift plan (+0.2m in robot Z).
            if bool(curobo_result.get("success", False)) and full_traj.shape[0] > 0:
                lift_target_eef_pose_robot = np.asarray(target_eef_pose_robot, dtype=np.float32).copy()
                lift_target_eef_pose_robot[2, 3] += 0.2
                lift_start_full_qpos = np.asarray(full_traj[-1], dtype=np.float32).reshape(-1)
                lift_start_qpos = lift_start_full_qpos[:6] if args.arm_only else lift_start_full_qpos
                lift_world = curobo_world
                if not args.curobo_ignore_world:
                    lift_world = {
                        "cuboid": dict(planning_scene["world"].get("cuboid", {})),
                        "mesh": {
                            k: v
                            for k, v in planning_scene["world"].get("mesh", {}).items()
                            if str(k) != "object"
                        },
                    }
                curobo_lift_result = _run_curobo_plan(
                    robot_cfg=robot_cfg,
                    world_cfg_dict=lift_world,
                    start_qpos=lift_start_qpos,
                    target_eef_pose_robot=lift_target_eef_pose_robot,
                    device=args.curobo_device,
                    require_dof=expected_active_dof,
                    seeds=int(args.curobo_seeds),
                    self_collision=(not bool(args.curobo_disable_self_collision)),
                )
                curobo_result["lift_after_grasp"] = {
                    "target_eef_pose_robot": lift_target_eef_pose_robot.tolist(),
                    "z_offset_m": 0.2,
                    "success": bool(curobo_lift_result.get("success", False)),
                    "status": curobo_lift_result.get("status"),
                    "error": curobo_lift_result.get("error"),
                }
                curobo_result["lift_after_grasp"]["world_has_object"] = False
                lift_full_traj = _build_full_traj_from_curobo_result(
                    curobo_result=curobo_lift_result,
                    plan_start_qpos=lift_start_full_qpos,
                    full_joint_names=list(robot_module.get_joint_names()),
                    arm_only=bool(args.arm_only),
                )
                if lift_full_traj is not None and lift_full_traj.shape[0] > 0:
                    # Drop the first point to avoid duplicating the pre-lift state.
                    if lift_full_traj.shape[0] > 1:
                        full_traj = np.concatenate([full_traj, lift_full_traj[1:]], axis=0)
                    curobo_result["lift_after_grasp"]["trajectory_added_steps"] = int(max(lift_full_traj.shape[0] - 1, 0))
                else:
                    curobo_result["lift_after_grasp"]["trajectory_added_steps"] = 0
            curobo_result["trajectory"] = full_traj.tolist()
            curobo_result["trajectory_full"] = full_traj.tolist()
            curobo_result["trajectory_post_hand_interp_steps"] = int(args.hand_post_interp_steps)
        if (
            (not curobo_result.get("success", False))
            and (not args.curobo_ignore_world)
            and ("IK_FAIL" in str(curobo_result.get("status", "")) or "IK_FAIL" in str(curobo_result.get("error", "")))
        ):
            diag = _run_curobo_plan(
                robot_cfg=robot_cfg,
                world_cfg_dict={"cuboid": {}, "mesh": {}},
                start_qpos=curobo_start_qpos,
                target_eef_pose_robot=target_eef_pose_robot,
                device=args.curobo_device,
                require_dof=expected_active_dof,
                seeds=int(args.curobo_seeds),
                self_collision=(not bool(args.curobo_disable_self_collision)),
            )
            curobo_result["diagnostic_no_world"] = {
                "success": bool(diag.get("success", False)),
                "status": diag.get("status"),
                "error": diag.get("error"),
            }

        curobo_json_path = os.path.join(abs_save_path, "curobo_plan_result.json")
        with open(curobo_json_path, "w", encoding="utf-8") as f:
            json.dump(curobo_result, f, ensure_ascii=False, indent=2)
        print(f"saved: {curobo_json_path}")

    if args.no_vis:
        return

    mesh = _load_mesh(mesh_path)
    vis = ViserViewer()
    vis.add_floor(height=0.0)
    vis.server.scene.add_box(
        "/world/table",
        dimensions=tuple(float(x) for x in table_size_xyz),
        position=tuple(float(x) for x in table_pos_xyz),
        color=(160, 130, 100),
    )
    vis.add_robot("robot", robot_urdf_path)
    _add_cameras_to_viser(vis, abs_save_path, c2r)
    vis.add_object(args.mesh_name, mesh, pose_robot, opacity=1.0)
    full_joint_names = list(robot_module.get_joint_names())
    robot_module.update_cfg(plan_start_qpos)
    start_eef_pose_robot = _to_4x4(
        robot_module.get_transform(args.eef_link, robot_module.urdf.base_link, collision_geometry=False)
    )
    start_eef_q = _rotmat_to_wxyz(start_eef_pose_robot[:3, :3])
    vis.server.scene.add_frame(
        "/target/start_eef_pose",
        position=start_eef_pose_robot[:3, 3].astype(float),
        wxyz=start_eef_q.astype(float),
        show_axes=True,
        axes_length=0.07,
        axes_radius=0.0035,
    )
    eef_q = _rotmat_to_wxyz(target_eef_pose_robot[:3, :3])
    vis.server.scene.add_frame(
        "/target/eef_pose",
        position=target_eef_pose_robot[:3, 3].astype(float),
        wxyz=eef_q.astype(float),
        show_axes=True,
        axes_length=0.08,
        axes_radius=0.004,
    )

    trajectory_for_mujoco: Optional[np.ndarray] = None
    if curobo_result is not None and curobo_result.get("success") and curobo_result.get("trajectory") is not None:
        traj = np.asarray(curobo_result["trajectory"], dtype=np.float32)
        active_joint_names = list(curobo_result.get("active_joint_names", []) or [])
        if traj.shape[-1] == plan_start_qpos.shape[0]:
            vis.add_traj("curobo_traj", {"robot": traj}, {})
            trajectory_for_mujoco = traj
        elif args.arm_only and traj.shape[-1] == 6:
            full_traj = np.stack([_expand_arm_qpos_to_full(q, plan_start_qpos) for q in traj], axis=0)
            vis.add_traj("curobo_traj", {"robot": full_traj}, {})
            trajectory_for_mujoco = full_traj
        else:
            full_traj = np.stack(
                [
                    _expand_active_qpos_to_full(q, plan_start_qpos, full_joint_names, active_joint_names)
                    for q in traj
                ],
                axis=0,
            )
            vis.add_traj("curobo_traj", {"robot": full_traj}, {})
            trajectory_for_mujoco = full_traj
    else:
        vis.robot_dict["robot"].update_cfg(plan_start_qpos)
        if curobo_result is not None and isinstance(curobo_result.get("ik_debug"), dict):
            ik_dbg = curobo_result["ik_debug"]
            best_q = np.asarray(ik_dbg.get("best_qpos", []), dtype=np.float32).reshape(-1)
            if best_q.size > 0:
                if args.arm_only and best_q.shape[0] == 6:
                    best_q_full = _expand_arm_qpos_to_full(best_q, plan_start_qpos)
                elif best_q.shape[0] == plan_start_qpos.shape[0]:
                    best_q_full = best_q
                else:
                    active_joint_names = list(curobo_result.get("active_joint_names", []) or [])
                    best_q_full = _expand_active_qpos_to_full(
                        best_q,
                        plan_start_qpos,
                        full_joint_names,
                        active_joint_names,
                    )
                if best_q_full.shape[0] == plan_start_qpos.shape[0]:
                    vis.add_robot("robot_ik_best", robot_urdf_path)
                    vis.robot_dict["robot_ik_best"].update_cfg(best_q_full)

    print("Viser running. Press Ctrl+C to exit.")
    try:
        while True:
            vis.update()
    except KeyboardInterrupt:
        pass

    if args.mujoco_after_vis:
        if trajectory_for_mujoco is None:
            print("[WARN] skip MuJoCo replay: no successful trajectory.")
        else:
            first_waypoint_err = float(np.linalg.norm(trajectory_for_mujoco[0] - plan_start_qpos))
            print(f"MuJoCo start check: ||traj[0]-plan_start_qpos||={first_waypoint_err:.6f}")
            print("Starting MuJoCo replay...")
            _run_mujoco_replay(
                robot_mjcf_path=os.path.expanduser(args.mujoco_robot_mjcf),
                world_dict=planning_scene["world"],
                trajectory=trajectory_for_mujoco,
                full_joint_names=full_joint_names,
                waypoint_hold_steps=int(args.mujoco_waypoint_hold_steps),
                object_mass=float(args.mujoco_object_mass),
                dynamic_step=bool(args.mujoco_dynamic_step),
                validate_lift=bool(args.mujoco_validate_lift),
                lift_success_z=float(args.mujoco_lift_success_z),
                start_settle_steps=int(args.mujoco_start_settle_steps),
                hand_kinematic_in_dynamic=bool(args.mujoco_hand_kinematic_in_dynamic),
                hand_use_position_actuator=(
                    bool(args.mujoco_hand_position_actuator)
                    and (not bool(args.mujoco_general_hand_actuator))
                ),
                actuator_use_xarm7_profile=(not bool(args.mujoco_no_xarm7_profile)),
                actuator_kp_arm=float(args.mujoco_arm_kp),
                actuator_kd_arm=float(args.mujoco_arm_kd),
                actuator_force_arm=float(args.mujoco_arm_force),
                actuator_kp_hand=float(args.mujoco_hand_kp),
                actuator_kd_hand=float(args.mujoco_hand_kd),
                actuator_force_hand=float(args.mujoco_hand_force),
                hand_ctrl_max_step=float(args.mujoco_hand_ctrl_max_step),
                hand_position_kp=float(args.mujoco_hand_position_kp),
                decimation=int(args.mujoco_decimation),
            )


if __name__ == "__main__":
    main()
