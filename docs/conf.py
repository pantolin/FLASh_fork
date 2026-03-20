# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root to sys.path to import the library directly.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "FLASh"
copyright = "2026, FLASh authors"
author = "Gonzalo Bonilla"

# The full version, including alpha/beta/rc tags
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx_rtd_dark_mode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "style_nav_header_background": "#1976d2",
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
