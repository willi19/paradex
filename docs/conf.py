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
    'myst_parser',              # Markdown pages (docs/*.md)
    'sphinxcontrib.mermaid',    # diagrams
    'sphinx_design',            # dropdowns/toggles, cards, tabs, grids
]

# MyST features used in the Markdown guides.
# colon_fence enables the ::: {dropdown} / ::: {note} block syntax.
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'fieldlist',                # :param:/:returns:/:rtype: → Parameters/Returns
    'attrs_inline',
    'substitution',
    'tasklist',
]

# Pin mermaid so the rendered diagrams are stable on the static site.
mermaid_version = '10.9.1'

autoapi_dirs = ['../paradex']
autoapi_type = 'python'
autoapi_python_use_implicit_namespaces = True

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