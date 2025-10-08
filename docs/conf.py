import os
import sys

project = 'paradex'
copyright = '2025, Mingi Choi'
author = 'Mingi Choi'
release = '0.0.0'

# 자동 문서화 확장 추가
extensions = [
    'sphinx.ext.autodoc',      # 자동 문서화
    'sphinx.ext.napoleon',     # Google/NumPy 스타일 docstring 지원
    'sphinx.ext.viewcode',     # 소스코드 링크
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'  # 'sphinx_rtd_theme' 추천!
html_static_path = ['_static']