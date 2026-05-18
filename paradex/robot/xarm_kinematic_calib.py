"""
Read per-unit kinematic calibration from xArm controller and apply to URDF.

For xArm/UF850 produced after August 2023, UFACTORY performs per-unit
kinematic calibration at the factory and stores the resulting offsets in
the controller register. Reading these values via TCP/Modbus (port 502)
and writing them into the URDF joint origins makes URDF FK match the
robot's actual end-effector pose.

Protocol matches xarm_ros2/xarm_description/config/kinematics/gen_kinematics_params.py.
"""
import os
import shutil
import socket
import struct
import xml.etree.ElementTree as ET

import yaml


_JOINT_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
_DEFAULT_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]


def _resolve_robot_name(dof: int, robot_type: int) -> str:
    if dof == 6 and robot_type == 12:
        return "uf850"
    if dof == 6 and robot_type == 9:
        return "lite6"
    return f"xarm{dof}"


def read_xarm_kinematic_params(robot_ip: str, timeout: float = 3.0) -> dict:
    """Connect to xArm controller and read factory kinematic calibration.

    Returns a dict shaped like::

        {
            "robot_dof": 6,
            "robot_name": "xarm6",
            "kinematics": {
                "joint1": {"x": .., "y": .., "z": .., "roll": .., "pitch": .., "yaw": ..},
                ...
            },
        }
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((robot_ip, 502))
        sock.send(bytes([0x00, 0x01, 0x00, 0x02, 0x00, 0x01, 0x08]))
        recv = sock.recv(179)
    finally:
        sock.close()

    if len(recv) != 179 or not recv[8]:
        raise RuntimeError(
            f"xArm kinematic readout failed (len={len(recv)}, "
            f"flag={recv[8] if len(recv) > 8 else 'n/a'}). "
            "Check robot IP, that the controller is running, and that the "
            "unit was produced after Aug 2023 (older units have no factory "
            "calibration register)."
        )

    dof = recv[9]
    robot_type = recv[10]
    floats = struct.unpack("<42f", recv[11:])

    kinematics = {}
    for i in range(dof):
        kinematics[f"joint{i + 1}"] = {
            "x": floats[i * 6 + 0],
            "y": floats[i * 6 + 1],
            "z": floats[i * 6 + 2],
            "roll": floats[i * 6 + 3],
            "pitch": floats[i * 6 + 4],
            "yaw": floats[i * 6 + 5],
        }

    return {
        "robot_dof": dof,
        "robot_name": _resolve_robot_name(dof, robot_type),
        "kinematics": kinematics,
    }


def save_kinematic_yaml(params: dict, path: str) -> None:
    """Save calibration in the xarm_ros2-compatible YAML format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"kinematics": params["kinematics"]}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)


def load_kinematic_yaml(path: str) -> dict:
    """Load YAML and return the joint -> {x,y,z,roll,pitch,yaw} mapping."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["kinematics"]


def apply_kinematics_to_urdf(
    urdf_path: str,
    kinematic_params: dict,
    joint_names=None,
) -> dict:
    """Patch joint origins in a URDF in-place.

    On first call, backs up the original to ``{urdf_path}.original``. On
    every call, parses the backup (not the live file) so re-running is
    idempotent and never compounds edits.

    Returns a diff summary: {joint_name: {"d_xyz_mm": [..], "d_rpy_deg": [..]}}.
    """
    import math

    if joint_names is None:
        joint_names = _DEFAULT_JOINT_NAMES

    backup_path = urdf_path + ".original"
    if not os.path.exists(backup_path):
        shutil.copy2(urdf_path, backup_path)

    tree = ET.parse(backup_path)
    root = tree.getroot()

    diff = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in joint_names:
            continue
        if name not in kinematic_params:
            continue
        kp = kinematic_params[name]
        origin = joint.find("origin")
        if origin is None:
            origin = ET.SubElement(joint, "origin")

        old_xyz = [float(v) for v in (origin.get("xyz") or "0 0 0").split()]
        old_rpy = [float(v) for v in (origin.get("rpy") or "0 0 0").split()]
        new_xyz = [kp["x"], kp["y"], kp["z"]]
        new_rpy = [kp["roll"], kp["pitch"], kp["yaw"]]

        origin.set("xyz", " ".join(f"{v:.10g}" for v in new_xyz))
        origin.set("rpy", " ".join(f"{v:.10g}" for v in new_rpy))

        diff[name] = {
            "d_xyz_mm": [(n - o) * 1000.0 for n, o in zip(new_xyz, old_xyz)],
            "d_rpy_deg": [math.degrees(n - o) for n, o in zip(new_rpy, old_rpy)],
        }

    tree.write(urdf_path, encoding="utf-8", xml_declaration=False)
    return diff
