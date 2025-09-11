import numpy as np
import trimesh
import os

from paradex.visualization.viser_viewer import ViserViewer
from paradex.utils.file_io import rsc_path, get_robot_urdf_path
from paradex.robot.robot_wrapper import RobotWrapper

robot = RobotWrapper(os.path.join(rsc_path, "robot", "xarm_allegro.urdf"))

obj_mesh = trimesh.load("pickplace/object/0/start.ply")# trimesh.load(os.path.join(rsc_path ,"object" ,"pringles", "pringles.obj"))
qpos = []
for obj_id in os.listdir("pickplace/traj"):
    qpos.append(np.load(f"pickplace/traj/{obj_id}/start_qpos.npy"))
    qpos.append(np.load(f"pickplace/traj/{obj_id}/pick_qpos.npy"))
    qpos.append(np.load(f"pickplace/traj/{obj_id}/end_qpos.npy"))
    qpos.append(np.load(f"pickplace/traj/{obj_id}/place_qpos.npy"))

qpos = np.concatenate(qpos, axis=0)
obj_T = np.array([np.eye(4) for _ in range(len(qpos))])

urdf_path = get_robot_urdf_path("xarm", "allegro")
viewer = ViserViewer(obj_mesh, obj_T, urdf_path, qpos)
viewer.start_viewer()