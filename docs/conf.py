import os, sys
sys.path.insert(0, os.path.abspath('..'))  # adjust if your package lives elsewhere

project = "Deep LVPM"
author = "Alex James Ing"
extensions = []  # add 'sphinx.ext.autodoc', 'sphinx.ext.napoleon' later if you want API docs
templates_path = ['_templates']
exclude_patterns = []
html_theme = 'furo'
html_static_path = ['_static']

html_logo = "../dlvpm_logo_final.png" 