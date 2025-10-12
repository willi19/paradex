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
    'trimesh', 'open3d', 'pyrender', 'curobo'
]

exclude_patterns = ['_build']

html_theme = 'furo'