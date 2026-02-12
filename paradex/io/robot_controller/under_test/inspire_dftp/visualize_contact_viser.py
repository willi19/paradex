import os
import time
import copy
import threading
import numpy as np
import trimesh
import viser
from typing import Dict
from pymodbus.client import ModbusTcpClient
from threading import Lock

from paradex.robot.robot_wrapper_deprecated import RobotWrapper
from paradex.robot.robot_module import robot_info

# ======================================================
# Path
# ======================================================
URDF_PATH = "/home/temp_id/paradex/rsc/robot/inspire_left.urdf"

# ======================================================
# Modbus config
# ======================================================
MODBUS_IP = "192.168.11.210"
MODBUS_PORT = 6000
TOUCH_SENSOR_BASE_ADDR = 3000
TOTAL_REG_COUNT = 2124

# ======================================================
# Tactile layout
# ======================================================
SENSOR_LAYOUT = {
    "little_tip":    {"addr": 3000, "rows": 3, "cols": 3},
    "little_nail":   {"addr": 3018, "rows": 12, "cols": 8},
    "little_pad":    {"addr": 3210, "rows": 10, "cols": 8},
    "ring_tip":      {"addr": 3370, "rows": 3, "cols": 3},
    "ring_nail":     {"addr": 3388, "rows": 12, "cols": 8},
    "ring_pad":      {"addr": 3580, "rows": 10, "cols": 8},
    "middle_tip":    {"addr": 3740, "rows": 3, "cols": 3},
    "middle_nail":   {"addr": 3758, "rows": 12, "cols": 8},
    "middle_pad":    {"addr": 3950, "rows": 10, "cols": 8},
    "index_tip":     {"addr": 4110, "rows": 3, "cols": 3},
    "index_nail":    {"addr": 4128, "rows": 12, "cols": 8},
    "index_pad":     {"addr": 4320, "rows": 10, "cols": 8},
    "thumb_tip":     {"addr": 4480, "rows": 3, "cols": 3},
    "thumb_nail":    {"addr": 4498, "rows": 12, "cols": 8},
    "thumb_middle":  {"addr": 4690, "rows": 3, "cols": 3},
    "thumb_pad":     {"addr": 4708, "rows": 12, "cols": 8},
    "palm":          {"addr": 4900, "rows": 8, "cols": 14},
}

# ======================================================
# Tactile → (link, vertex ids)
# ======================================================
TACTILE_VERTEX_MAP = {
    "little_tip":    ("pinky_intermediate", [5006, 5191, 5045, 5041]),
    "little_nail":   ("pinky_intermediate", [5207, 5481, 5380, 5246]),
    "little_pad":    ("pinky_proximal",     [4816, 4820, 4803, 4806]),

    "ring_tip":      ("ring_intermediate",  [3845, 3843, 4265, 4260]),
    "ring_nail":     ("ring_intermediate",  [3964, 4080, 4368, 4458]),
    "ring_pad":      ("ring_proximal",      [4820, 4816, 4806, 4803]),

    "middle_tip":    ("middle_intermediate",[3736, 3738, 4118, 4121]),
    "middle_nail":   ("middle_intermediate",[4171, 4265, 3883, 3789]),
    "middle_pad":    ("middle_proximal",    [4822, 4815, 4819, 4824]),

    "index_tip":     ("index_intermediate", [3843, 3976, 4379, 4265]),
    "index_nail":    ("index_intermediate", [4458, 4471, 4094, 4080]),
    "index_pad":     ("index_proximal",     [4822, 4815, 4819, 4824]),

    "thumb_tip":     ("thumb_distal",       [713, 706, 667, 677]),
    "thumb_nail":    ("thumb_distal",       [992, 993, 941, 942]),
    "thumb_middle":  ("thumb_proximal",     [235, 236, 237, 240]),
    "thumb_pad":     ("thumb_proximal",     [554, 389, 371, 532]),

    "palm":          ("hand_base_link",     [25780, 11619, 26082, 26141]),
}

# ======================================================
# Shared tactile scalar
# ======================================================
tactile_scalar: Dict[str, float] = {}
tactile_lock = Lock()

# ======================================================
# Modbus utils
# ======================================================
def parse_tactile_data(registers):
    parsed = {}
    for name, meta in SENSOR_LAYOUT.items():
        start = meta["addr"] - TOUCH_SENSOR_BASE_ADDR
        rows, cols = meta["rows"], meta["cols"]
        raw = registers[start:start + rows * cols]
        parsed[name] = np.array(raw).reshape(rows, cols)
    return parsed

def modbus_thread():
    client = ModbusTcpClient(MODBUS_IP, port=MODBUS_PORT)
    client.connect()

    while True:
        regs = []
        for addr in range(TOUCH_SENSOR_BASE_ADDR, 5124, 120):
            cnt = min(120, 5124 - addr)
            res = client.read_holding_registers(addr, cnt)
            if not res.isError():
                regs.extend(res.registers)

        if len(regs) == TOTAL_REG_COUNT:
            tactile_map = parse_tactile_data(regs)
            with tactile_lock:
                for k, v in tactile_map.items():
                    tactile_scalar[k] = float(np.mean(v))

        time.sleep(0.03)

# ======================================================
# Robot mesh utils
# ======================================================
def get_mesh(robot_wrapper: RobotWrapper, qpos: np.ndarray):
    robot_wrapper.compute_forward_kinematics(qpos)
    robot_obj = robot_info(URDF_PATH, down_sample=True)

    mesh_vis = []
    for link_nm, mesh_items in robot_obj.mesh_dict.items():
        if not mesh_items:
            continue

        link_pose = robot_wrapper.get_link_pose(
            robot_wrapper.get_link_index(link_nm)
        )

        for o3d_mesh in mesh_items:
            mesh = copy.deepcopy(o3d_mesh)
            mesh.transform(link_pose)

            tm = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles),
                process=False,
            )
            mesh_vis.append((link_nm, tm))
    return mesh_vis

# ======================================================
# Arrow geometry
# ======================================================

def length_to_color(length, max_len=0.025):
    """
    length: 화살표 길이
    return: (R, G, B, A) in uint8
    """
    t = np.clip(length / max_len, 0.0, 1.0)

    # Blue → Cyan → Green → Yellow → Red
    if t < 0.25:
        r, g, b = 0, int(4*t*255), 255
    elif t < 0.5:
        r, g, b = 0, 255, int((1-4*(t-0.25))*255)
    elif t < 0.75:
        r, g, b = int(4*(t-0.5)*255), 255, 0
    else:
        r, g, b = 255, int((1-4*(t-0.75))*255), 0

    return np.array([r, g, b, 255], dtype=np.uint8)


def compute_contact_arrow(tm: trimesh.Trimesh, vertex_ids):
    v = tm.vertices[vertex_ids]
    n = tm.vertex_normals[vertex_ids]
    center = v.mean(axis=0)
    normal = n.mean(axis=0)
    normal /= np.linalg.norm(normal) + 1e-8
    return center, normal

def make_arrow_mesh(start, direction, length):
    shaft = trimesh.creation.cylinder(radius=0.001, height=length * 0.7)
    head = trimesh.creation.cone(radius=0.002, height=length * 0.3)
    shaft.apply_translation([0, 0, length * 0.35])
    head.apply_translation([0, 0, length * 0.85])
    arrow = trimesh.util.concatenate([shaft, head])
    T = trimesh.geometry.align_vectors([0, 0, 1], direction)
    arrow.apply_transform(T)
    arrow.apply_translation(start)
    return arrow

# ======================================================
# Main
# ======================================================
def main():
    server = viser.ViserServer()
    robot_wrapper = RobotWrapper(URDF_PATH)

    # start tactile thread
    threading.Thread(target=modbus_thread, daemon=True).start()

    robot_mesh_handles: Dict[str, viser.MeshHandle] = {}
    arrow_mesh_handles: Dict[str, viser.MeshHandle] = {}

    qpos = np.zeros(12, dtype=np.float32)

    MAX_PRESSURE = 1000.0
    MAX_ARROW_LEN = 0.025

    while True:
        mesh_vis = get_mesh(robot_wrapper, qpos)

        # ---------------- robot mesh ----------------
        for link_name, tm in mesh_vis:
            name = f"/robot/{link_name}"
            if name in robot_mesh_handles:
                robot_mesh_handles[name].remove()
            robot_mesh_handles[name] = server.scene.add_mesh_trimesh(
                name=name,
                mesh=tm,
            )

        # snapshot tactile
        with tactile_lock:
            tactile_snapshot = tactile_scalar.copy()

        # ---------------- arrows ----------------
        for sensor, (link, v_ids) in TACTILE_VERTEX_MAP.items():
            tm = next((m for ln, m in mesh_vis if ln == link), None)
            if tm is None:
                continue

            pressure = tactile_snapshot.get(sensor, 0.0)
            length = np.clip(pressure / MAX_PRESSURE, 0.0, 1.0) * MAX_ARROW_LEN

            center, normal = compute_contact_arrow(tm, v_ids)
            length = np.clip(p / 1000.0, 0, 1) * 0.025
            color = length_to_color(length, max_len=0.025)
            
            arrow_mesh = make_arrow_mesh(center, normal, length)

            name = f"/contact_arrow/{sensor}"
            if name in arrow_mesh_handles:
                arrow_mesh_handles[name].remove()
            arrow_mesh_handles[name] = server.scene.add_mesh_trimesh(
                name=name,
                mesh=arrow_mesh,
            )

        time.sleep(0.03)

if __name__ == "__main__":
    main()
