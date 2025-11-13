from paradex.visualization.visualizer.viser import ViserViewer
from paradex.utils.file_io import get_robot_urdf_path

a = ViserViewer()
a.add_robot("franka", get_robot_urdf_path("franka"))
a.start_viewer()