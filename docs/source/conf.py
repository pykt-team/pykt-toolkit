# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../pykt/'))


# -- Project information -----------------------------------------------------

project = 'pykt-toolkit'
copyright = '2022, pykt-team'
author = 'pykt-team'

# The full version, including alpha/beta/rc tags
release = '0.0.37'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc']

source_suffix = ['.rst', '.md']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
exclude_patterns = []




# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'

# # Add any paths that contain custom static files (such as style sheets) here,
# # relative to this directory. They are copied after the builtin static files,
# # so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# theme
extensions.append('sphinx_rtd_theme')
html_theme = "sphinx_rtd_theme"
html_logo = "https://pykt.org/assets/images/logo.png"
html_theme_options = {
    'logo_only': True,
    'navigation_depth': 5,
    'display_version': True,
}


# markdown
source_parsers = {
    '.md': 'recommonmark.parser.CommonMarkParser',
}
extensions.append('recommonmark')

# autoapi-python
# extensions.append('autoapi.extension')
extensions.append("sphinx.ext.napoleon")
# autoapi_type = 'python'
# autoapi_dirs = ['../../pykt']
# autoapi_options = ['members', 'undoc-members', 'show-inheritance', 'show-module-summary', 'imported-members']
# autoapi_add_toctree_entry = False

autodoc_mock_imports = [
]
