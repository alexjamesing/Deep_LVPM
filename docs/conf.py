import os, sys
sys.path.insert(0, os.path.abspath('..'))  # adjust if your package lives elsewhere

project = "Deep LVPM"
author = "Your Name"
extensions = []  # add 'sphinx.ext.autodoc', 'sphinx.ext.napoleon' later if you want API docs
templates_path = ['_templates']
exclude_patterns = []
html_theme = 'furo'
html_static_path = ['_static']

# Optional: show a logo (put the file at docs/_static/dlvpm_logo_final.png)
html_logo = "/Users/ing/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents_mac_korbel33/Github/DLVPM/DLVPM/Gitlab_upload/Deep_LVPM2/dlvpm_logo_final.png"