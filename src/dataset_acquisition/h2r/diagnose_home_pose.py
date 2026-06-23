"""
Visualize the planning start state used by capture_object6d_allegro_v5.py so
we can see which collision sphere intersects the table or object.

Loads the most recent saved 6D pose for an object/scene, adds the xarm +
allegro_v5 robot at HOME pose (arm = HOME_DEG, hand pre-shaped to canonical
wp0), draws the table cuboid + object mesh, and overlays every robot collision
sphere as a wireframe so the offender is obvious.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).parents[3]
sys.path.append(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paradex.visualization.robot import RobotModule
from paradex.visualization.visualizer.viser import ViserViewer

HOME_DEG = [-84.9, -25.5, -41.1, 118.8, -1.9, 158.1]
ALLEGRO_HAND_JOINTS = [f"joint_{i}_0" for i in range(16)]


def _load_pose_robot(save_root: Path, mesh_name: str, scene: int, src: str):
    base = save_root / mesh_name / str(scene) / src
    if not base.is_dir():
        raise FileNotFoundError(base)
    runs = sorted(p for p in base.iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"no run dirs under {base}")
    j = json.loads((runs[-1] / "object_6d_robot.json").read_text())
    return np.asarray(j["pose_robot"], dtype=np.float32), runs[-1]


def _link_world_poses(robot_urdf, qpos, joint_names):
    """Use the existing RobotModule + trimesh-based forward kinematics to get
    each link's world pose at the given qpos."""
    import xml.etree.ElementTree as ET
    from collections import defaultdict

    tree = ET.parse(robot_urdf)
    root = tree.getroot()
    parent_of, joint_of_link = {}, {}
    joints = {}
    for j in root.findall("joint"):
        name = j.attrib["name"]
        jtype = j.attrib["type"]
        parent = j.find("parent").attrib["link"]
        child = j.find("child").attrib["link"]
        origin = j.find("origin")
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        axis = j.find("axis")
        axis_xyz = np.fromstring(axis.attrib["xyz"], sep=" ") if axis is not None else np.array([0, 0, 1.0])
        joints[name] = dict(type=jtype, parent=parent, child=child, xyz=xyz, rpy=rpy, axis=axis_xyz)
        parent_of[child] = parent
        joint_of_link[child] = name

    qmap = {n: float(qpos[i]) for i, n in enumerate(joint_names)}
    cache = {}

    def world(link):
        if link in cache:
            return cache[link]
        if link not in joint_of_link:
            cache[link] = np.eye(4)
            return cache[link]
        jn = joint_of_link[link]
        jd = joints[jn]
        T = np.eye(4)
        T[:3, :3] = R.from_euler("xyz", jd["rpy"]).as_matrix()
        T[:3, 3] = jd["xyz"]
        if jd["type"] == "revolute" and jn in qmap:
            Rq = R.from_rotvec(jd["axis"] * qmap[jn]).as_matrix()
            Tq = np.eye(4); Tq[:3, :3] = Rq
            T = T @ Tq
        parent_T = world(jd["parent"])
        cache[link] = parent_T @ T
        return cache[link]

    return world


def _add_spheres_overlay(vis, spheres_yml: Path, robot_urdf: Path, qpos, joint_names,
                         table_z_top: float):
    spheres = yaml.safe_load(spheres_yml.read_text())["collision_spheres"]
    world = _link_world_poses(str(robot_urdf), qpos, joint_names)
    server = vis.server
    n_hit = 0
    for link_name, slist in spheres.items():
        T = world(link_name)
        for i, s in enumerate(slist):
            c = np.array(s["center"], dtype=np.float32)
            r = float(s["radius"])
            cw = (T[:3, :3] @ c + T[:3, 3])
            hits_table = (cw[2] - r) < table_z_top
            color = (255, 80, 80) if hits_table else (120, 200, 255)
            if hits_table:
                n_hit += 1
            server.scene.add_icosphere(
                name=f"/spheres/{link_name}__{i}",
                radius=r,
                position=tuple(float(x) for x in cw),
                color=color,
            )
    print(f"[diagnose] {n_hit} sphere(s) penetrate table top (z<{table_z_top})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--src", default="robot")
    ap.add_argument("--save_root",
                    default="/home/temp_id/shared_data/capture/test_allegro")
    ap.add_argument("--legacy_save_root",
                    default="/home/temp_id/shared_data/capture/h2r/object_6d_allegro_v5",
                    help="fallback location for older runs")
    ap.add_argument("--mesh_root_dir",
                    default="/home/temp_id/shared_data/mesh_blender")
    ap.add_argument("--result_root",
                    default="/home/temp_id/shared_data/capture/h2r/0518")
    ap.add_argument("--table_size_xyz", default="2.4,1.6,0.1")
    ap.add_argument("--table_pos_xyz", default="0.0,0.0,-0.05")
    ap.add_argument("--robot_urdf",
                    default=str(PROJECT_ROOT / "rsc/robot/xarm_allegro_v5.urdf"))
    ap.add_argument("--spheres_yml",
                    default=str(PROJECT_ROOT / "rsc/curobo/content/configs/robot/spheres/xarm_allegro_v5.yml"))
    args = ap.parse_args()

    # 1) find pose_robot from a prior run
    try:
        pose_robot, run_dir = _load_pose_robot(Path(args.save_root), args.name, args.scene, args.src)
    except FileNotFoundError:
        pose_robot, run_dir = _load_pose_robot(Path(args.legacy_save_root), args.name, args.scene, args.src)
    print(f"[diagnose] pose_robot loaded from {run_dir}")
    print(f"[diagnose] object t = {pose_robot[:3,3].round(4).tolist()}")

    # 2) start qpos: arm = HOME_DEG, hand = canonical wp0 fingers
    rm = RobotModule(args.robot_urdf)
    jn = rm.get_joint_names()
    qpos = np.zeros(rm.get_num_joints(), dtype=np.float32)
    qpos[:6] = np.deg2rad(np.asarray(HOME_DEG, dtype=np.float32))

    res_path = Path(args.result_root) / args.name / str(args.scene) / "result_allegro.npz"
    if res_path.exists():
        d = np.load(res_path, allow_pickle=True)
        traj0 = np.asarray(d[f"{args.src}_traj_q"])[0]
        for i, name in enumerate(ALLEGRO_HAND_JOINTS):
            if name in jn:
                qpos[jn.index(name)] = float(traj0[6 + i])
        print(f"[diagnose] hand pre-shaped from {res_path.name} wp0")
    else:
        print(f"[diagnose] {res_path} missing; hand stays at zeros")

    # 3) launch viser
    table_size = np.array([float(x) for x in args.table_size_xyz.split(",")])
    table_pos = np.array([float(x) for x in args.table_pos_xyz.split(",")])
    table_z_top = float(table_pos[2] + 0.5 * table_size[2])

    vis = ViserViewer()
    vis.add_floor(height=0.0)
    vis.add_robot("robot", args.robot_urdf)
    vis.robot_dict["robot"].update_cfg(qpos)

    # table cuboid (wireframe-ish: opaque cuboid is fine)
    vis.server.scene.add_box(
        name="/world/table",
        dimensions=tuple(table_size.tolist()),
        position=tuple(table_pos.tolist()),
        color=(180, 180, 180),
    )

    # object mesh
    mesh_path = Path(args.mesh_root_dir) / args.name / f"{args.name}_viser.obj"
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    vis.add_object(args.name, mesh, pose_robot, opacity=0.6)

    # collision sphere overlay; red = penetrating table
    _add_spheres_overlay(vis, Path(args.spheres_yml), Path(args.robot_urdf), qpos, jn, table_z_top)

    print(f"[diagnose] table top z = {table_z_top}")
    print("[diagnose] viser running. Open the URL in a browser. Ctrl+C to quit.")
    try:
        while True:
            vis.update()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
