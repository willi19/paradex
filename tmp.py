import os
from paradex.utils.file_io import rsc_path
import open3d as o3d
import numpy as np

obj_mesh = o3d.io.read_triangle_mesh(os.path.join(rsc_path, "bottle", "bottle.obj"))
vertices = np.asarray(obj_mesh.vertices)

import pdb; pdb.set_trace()