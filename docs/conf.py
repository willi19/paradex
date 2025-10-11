import os
import sys
sys.path.insert(0, os.path.abspath('..')) 

autodoc_mock_imports = [
    'transforms3d', 'cv2', 'torch', 'pyrender', 
    'trimesh', 'open3d', 'numpy'
]

project = 'paradex'
copyright = '2025, Mingi Choi'
author = 'Mingi Choi'
release = '0.0.0'

# -- General configuration
extensions = [
    'sphinx.ext.autodoc',       # 자동 문서화
    'sphinx.ext.autosummary',   # ← 추가! 자동 요약 생성
    'sphinx.ext.napoleon',      # Google/NumPy docstring
    'sphinx.ext.viewcode',      # 소스코드 링크
    'sphinx.ext.intersphinx',   # ← 추가! 다른 문서 링크
]

autosummary_generate = True  
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'show-inheritance': True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

templates_path = ['_templates']
exclude_patterns = [
    '_build',           # 빌드 출력 디렉토리
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
    'docs-html',        # ← 추가
]

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static'] 

html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'display_version': True,
}
