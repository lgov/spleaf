import os
import sys
import spleaf

# -- Project information -----------------------------------------------------
project = 'S+LEAF'
copyright = '2019, Jean-Baptiste Delisle'
author = 'Jean-Baptiste Delisle'

# -- General configuration ---------------------------------------------------
needs_sphinx = '1.1'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary',
              'sphinx.ext.intersphinx', 'sphinx.ext.coverage',
              'numpydoc']
templates_path = ['_templates']
exclude_patterns = []
pygments_style = 'sphinx'
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
html_theme_path = ['_theme']
html_theme = 'scipy'
html_theme_options = {
    "rootlinks": [],#[("http://www.unige.ch/", "UniGe")],
    'sidebar': 'right',
    "edit_link": False,
}
html_sidebars = {'index': ['searchbox.html']}
html_static_path = ['_static']