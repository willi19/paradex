import numpy as np
import trimesh
import os

from paradex.visualization.viser_viewer import ViserViewer
from paradex.utils.file_io import rsc_path, get_robot_urdf_path

obj_mesh = trimesh.load(os.path.join(rsc_path ,"object" ,"pringles", "pringles.obj"))
qpos = np.load("pickplace/traj.npy")[0]
obj_T = np.array([np.eye(4) for _ in range(len(qpos))])

urdf_path = get_robot_urdf_path("xarm", None)
viewer = ViserViewer(obj_mesh, obj_T, urdf_path, qpos)
viewer.start_viewer()