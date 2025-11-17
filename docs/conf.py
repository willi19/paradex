import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'paradex'
copyright = '2025, Mingi Choi'
author = 'Mingi Choi'
release = '0.0.0'

extensions = [
    'autoapi.extension',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

autoapi_dirs = ['../paradex']
autoapi_type = 'python'

# 문법 에러 있는 파일들 제외
autoapi_ignore = [
    '*/geometry/mesh.py',
    '*/visualization/viser_viewer_collision.py',
    '*/__pycache__/*',
    '*/tests/*',
]

autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
]

exclude_patterns = ['_build']

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']

html_logo = None
html_favicon = None

autosummary_generate = False