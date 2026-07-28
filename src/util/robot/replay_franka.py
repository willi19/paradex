"""Show the LIVE franka pose with the flat 5x6 board attached, in Viser.

Run on the franka PC with the daemon up (./cpp/franka_daemon/run_daemon.sh). Reads
the current qpos and draws franka + board_adapter plate + the flat 5x6 board where it
should be mounted, so you can check the physical board matches the model. Falls back
to the home pose if no live state (daemon down) so you can still eyeball the mount.

    python src/util/robot/replay_franka.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation as R

sys.path.append(str(Path(__file__).resolve().parents[2]))
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.robot.utils import get_robot_urdf_path
from paradex.utils.system import network_info

EEF_LINK = "fr3_link8"
BOARD_Z_FROM_EEF = 0.0475          # board plane along flange +z (board_adapter top + 3mm/2)
BOARD_W, BOARD_H, BOARD_T = 0.30, 0.36, 0.003
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def Tz(z):
    T = np.eye(4); T[2, 3] = z; return T


def main():
    urdf_path = get_robot_urdf_path(arm_name="franka")
    fk = yourdfpy.URDF.load(urdf_path, build_scene_graph=True)

    # connect to the franka daemon (plain IP string in network_info)
    franka = None
    try:
        from paradex.io.robot_controller.franka_controller import FrankaController
        franka = FrankaController(network_info["franka"])
        print("connected to franka daemon")
    except Exception as e:
        print(f"(no live franka: {e}) -> showing HOME pose")

    viewer = ViserViewer(port_number=8080)
    viewer.add_grid(size=2.0, cell_size=0.1, height=0.0)
    viewer.add_robot("franka", urdf_path)
    board = trimesh.creation.box(extents=[BOARD_W, BOARD_H, BOARD_T])
    board.visual.vertex_colors = [235, 235, 235, 255]
    viewer.add_trimesh("board", board, np.eye(4))

    def set_board(q):
        fk.update_cfg(q)
        Tb = np.array(fk.get_transform(EEF_LINK, fk.base_link)) @ Tz(BOARD_Z_FROM_EEF)
        h = viewer.obj_dict["board"]["frame"]
        h.position = Tb[:3, 3]
        h.wxyz = R.from_matrix(Tb[:3, :3]).as_quat()[[3, 0, 1, 2]]

    print("open http://localhost:8080 . live board mount overlay. Ctrl-C to quit.")
    try:
        while True:
            q = HOME
            if franka is not None:
                d = franka.get_data()
                if d is not None:
                    q = np.array(d["qpos"])
            viewer.robot_dict["franka"].update_cfg(q)
            set_board(q)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if franka is not None:
            try:
                franka.end()
            except Exception:
                pass


if __name__ == "__main__":
    main()
