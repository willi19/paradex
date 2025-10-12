import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'paradex'
copyright = '2025, Mingi Choi'
author = 'Mingi Choi'
release = '0.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

autodoc_mock_imports = [
    'PySpin', 'xarm', 'serial', 'rospy', 'sensor_msgs',
    'cv2', 'torch', 'numpy', 'transforms3d', 'scipy',
    'trimesh', 'open3d', 'pyrender', 'curobo',
    # 추가
    'pinocchio', 'zmq', 'pycolmap', 'plotly', 
    'nvdiffrast', 'tqdm',
]

exclude_patterns = ['_build']

html_theme = 'furo'

# 로고/파비콘 없음
html_logo = None
html_favicon = None

# 깔끔한 설정
html_theme_options = {
    "sidebar_hide_name": False,
}