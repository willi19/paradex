import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'paradex'
copyright = '2025, Mingi Choi'
author = 'Mingi Choi'
release = '0.0.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

# Mock imports
autodoc_mock_imports = [
    'transforms3d', 'cv2', 'torch', 'pyrender',
    'trimesh', 'open3d', 'numpy', 'PySpin',
    'xarm', 'serial', 'rospy', 'sensor_msgs', 'scipy',
]

# Autosummary
autosummary_generate = True

# Templates and static files
templates_path = ['_templates']
html_static_path = ['_static']
html_css_files = ['custom.css']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML theme
html_theme = 'furo'
html_logo = "_static/logo.png"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#7C4DFF",
        "color-brand-content": "#7C4DFF",
    },
}

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}