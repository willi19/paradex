"""View a saved hand-eye trajectory in Viser — INCLUDING the motion between waypoints.

Interpolates the joint-space straight line between consecutive waypoints (HOME -> wp0 ->
wp1 -> ...), the same path the daemon takes, so you can watch whether the board scrapes
the floor mid-move (joint limits may route it fine, or not). Each frame shows board_min_z;
the global minimum over the whole path is printed up front.

    python src/calibration/handeye/view_traj.py                 # latest hecalib/franka
    python src/calibration/handeye/view_traj.py --steps 25 --floor-z 0.037
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation as R

sys.path.append(str(Path(__file__).resolve().parents[3]))
from paradex.utils.system import config_dir
from paradex.calibration.utils import handeye_calib_path
from paradex.robot.utils import get_robot_urdf_path
from paradex.visualization.visualizer.viser import ViserViewer

def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ph = _load("ph", "plan_handeye_poses.py")   # load_cameras
pf = _load("pf", "plan_handeye_flat.py")     # board_min_z, BOARD_* constants

EEF_LINK = "fr3_link8"
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def Tz(z):
    T = np.eye(4); T[2, 3] = z; return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="franka")
    ap.add_argument("--cam-session", default=None)
    ap.add_argument("--floor-z", type=float, default=0.037)
    ap.add_argument("--steps", type=int, default=20, help="interpolation steps per segment")
    ap.add_argument("--no-home", action="store_true", help="don't prepend HOME as the start")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    traj_dir = os.path.join(config_dir, "hecalib", args.arm)
    files = sorted([f for f in os.listdir(traj_dir) if f.endswith("_qpos.npy")],
                   key=lambda f: int(f.split("_")[0]))
    wps = [np.load(os.path.join(traj_dir, f)) for f in files]
    print(f"{len(wps)} waypoints from {traj_dir}")
    if not wps:
        return
    seq = wps if args.no_home else [HOME] + wps

    urdf_path = get_robot_urdf_path(arm_name=args.arm)
    fk = yourdfpy.URDF.load(urdf_path, build_scene_graph=True)

    # dense joint-space path (straight line per segment) + waypoint markers
    dense, is_wp = [], []
    for i in range(len(seq) - 1):
        for k in range(args.steps):
            dense.append((1 - k / args.steps) * seq[i] + (k / args.steps) * seq[i + 1])
            is_wp.append(k == 0)
    dense.append(seq[-1]); is_wp.append(True)

    # board_min_z along the whole path
    def board_z(q):
        fk.update_cfg(q)
        Tb = np.array(fk.get_transform(EEF_LINK, fk.base_link)) @ Tz(pf.BOARD_Z_FROM_EEF)
        return pf.board_min_z(Tb), Tb
    bmins = [board_z(q)[0] for q in dense]
    gmin = min(bmins)
    print(f"path board_min_z over {len(dense)} frames: MIN {gmin:.3f} at frame {int(np.argmin(bmins))} "
          f"(floor {args.floor_z}; {'SCRAPES!' if gmin < args.floor_z else 'clears floor'})")

    viewer = ViserViewer(port_number=args.port)
    viewer.add_grid(size=2.0, cell_size=0.1, height=args.floor_z)
    viewer.add_robot(args.arm, urdf_path)
    board = trimesh.creation.box(extents=[pf.BOARD_W, pf.BOARD_H, pf.BOARD_T])
    board.visual.vertex_colors = [235, 235, 235, 255]
    viewer.add_trimesh("board", board, np.eye(4))

    session = args.cam_session or next(
        (s for s in sorted(os.listdir(handeye_calib_path), reverse=True)
         if os.path.exists(os.path.join(handeye_calib_path, s, "0", "C2R.npy"))), None)
    if session:
        for cam in ph.load_cameras(session):
            T = np.eye(4); T[:3, :3] = cam["Rwc"].T; T[:3, 3] = cam["C"]
            viewer.add_camera(cam["serial"], T, cam["K"], color=(0.0, 0.9, 0.3), size=0.05)

    sl = viewer.server.gui.add_slider("frame", min=0, max=len(dense) - 1, step=1, initial_value=0)
    play = viewer.server.gui.add_checkbox("play", False)
    txt = viewer.server.gui.add_text("info", initial_value="")

    def show(_=None):
        i = int(sl.value)
        viewer.robot_dict[args.arm].update_cfg(dense[i])
        _, Tb = board_z(dense[i])
        h = viewer.obj_dict["board"]["frame"]
        h.position = Tb[:3, 3]
        h.wxyz = R.from_matrix(Tb[:3, :3]).as_quat()[[3, 0, 1, 2]]
        flag = "  <-- BELOW FLOOR" if bmins[i] < args.floor_z else ""
        txt.value = f"frame {i}/{len(dense)-1} {'[waypoint]' if is_wp[i] else ''}  board_min_z={bmins[i]:.3f}{flag}"
    sl.on_update(show)
    show()

    print(f"open http://localhost:{args.port} . 'play' animates the full motion. Ctrl-C to quit.")
    try:
        while True:
            if play.value:
                sl.value = (int(sl.value) + 1) % len(dense)
                time.sleep(0.03)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
