"""Interactive tuner for the Franka + adapter + board-adapter stack in Viser.

Loads the BARE franka URDF and lays the two adapter meshes on the flange as movable
objects. GUI sliders adjust each piece's z offset (and the board's yaw) live; the
slider values ARE the URDF fixed-joint origins, so once it looks right, read them off
and I'll bake them into rsc/robot/franka/franka_adapter.urdf.

  adapter z  = adapter_joint  origin (fr3_link8 -> adapter)
  board z    = board_adapter_joint origin (adapter -> board_adapter)

Usage:
    python src/util/robot/view_franka_adapter.py
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

sys.path.append(str(Path(__file__).resolve().parents[3]))
from paradex.visualization.visualizer.viser import ViserViewer

REPO = Path(__file__).resolve().parents[3]
FRANKA = str(REPO / "rsc/robot/franka/franka.urdf")
ADAPTER_MESH = str(REPO / "rsc/adapter/Franka_base.obj")
BOARD_MESH = str(REPO / "rsc/adapter/Board_base.obj")
CUBE_MESH = str(REPO / "rsc/object/charuco_cube/charuco_cube.obj")


def Tz(z):
    T = np.eye(4)
    T[2, 3] = z
    return T


def Rz(deg):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("z", deg, degrees=True).as_matrix()
    return T


def flange_pose_zero_cfg():
    """fr3_link8 world pose at zero joint config (matches how add_robot displays it)."""
    import yourdfpy
    u = yourdfpy.URDF.load(FRANKA, build_scene_graph=True)
    u.update_cfg(np.zeros(len(u.actuated_joint_names)))
    return np.array(u.get_transform("fr3_link8", u.base_link))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    T_flange = flange_pose_zero_cfg()

    viewer = ViserViewer(port_number=args.port)
    viewer.add_grid(size=2.0, cell_size=0.1, height=0.0)
    viewer.add_robot("franka", FRANKA)

    adapter = trimesh.load(ADAPTER_MESH, force="mesh")
    board = trimesh.load(BOARD_MESH, force="mesh")
    cube = trimesh.load(CUBE_MESH, force="mesh")   # textured charuco cube
    adapter.visual.vertex_colors = [255, 100, 25, 255]
    board.visual.vertex_colors = [50, 130, 255, 255]
    viewer.add_trimesh("adapter", adapter, T_flange)
    viewer.add_trimesh("board", board, T_flange)
    viewer.add_trimesh("cube", cube, T_flange)

    def set_pose(name, T):
        h = viewer.obj_dict[name]["frame"]
        h.position = T[:3, 3]
        h.wxyz = R.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]

    with viewer.server.gui.add_folder("Adapter stack (z = URDF joint origins)"):
        s_adapter = viewer.server.gui.add_slider(
            "franka adapter z (m)", min=0.0, max=0.05, step=0.0005, initial_value=0.016)
        s_board = viewer.server.gui.add_slider(
            "board adapter z (m)", min=0.0, max=0.08, step=0.0005, initial_value=0.0)
        s_yaw = viewer.server.gui.add_slider(
            "board yaw (deg)", min=-180.0, max=180.0, step=1.0, initial_value=0.0)
        s_cube_x = viewer.server.gui.add_slider(
            "cube x (m)", min=-0.1, max=0.1, step=0.001, initial_value=0.0)
        s_cube_y = viewer.server.gui.add_slider(
            "cube y (m)", min=-0.1, max=0.1, step=0.001, initial_value=0.0)
        s_cube_z = viewer.server.gui.add_slider(
            "cube center z (m)", min=0.0, max=0.4, step=0.001, initial_value=0.18)
        readout = viewer.server.gui.add_text("current", initial_value="")

    def T_xyz(x, y, z):
        T = np.eye(4)
        T[:3, 3] = [x, y, z]
        return T

    def update(_=None):
        Ta = T_flange @ Tz(s_adapter.value)
        Tb = Ta @ Tz(s_board.value) @ Rz(s_yaw.value)
        Tc = Tb @ T_xyz(s_cube_x.value, s_cube_y.value, s_cube_z.value)
        set_pose("adapter", Ta)
        set_pose("board", Tb)
        set_pose("cube", Tc)
        readout.value = (f"adapter_z={s_adapter.value:.4f}  board_z={s_board.value:.4f}  "
                         f"yaw={s_yaw.value:.0f}  cube=({s_cube_x.value:.3f},{s_cube_y.value:.3f},{s_cube_z.value:.3f})")

    for s in (s_adapter, s_board, s_yaw, s_cube_x, s_cube_y, s_cube_z):
        s.on_update(update)
    update()

    print("Drag the sliders. When it looks right, read the 'current' values and give them to me.")
    print("(adapter=orange, board=blue). Ctrl-C to quit.")
    viewer.start_viewer(use_thread=False)


if __name__ == "__main__":
    main()
