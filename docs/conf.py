# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
import os

project = 'Hespas'
copyright = '2026 imec'
author = 'imec'
release = '0.1'
root_doc = "index"

autodoc_mock_imports = [
    "pandas", "plotly", "tqdm", "mlir", "iree", "tensorflow", "torch", "jax",
    "onnx", "scalesim",
    "hespas.estimator.iree_estimator",
    "hespas.estimator.onnxim_estimator",
]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx_rtd_theme', 'sphinxcontrib.bibtex',
              'sphinx.ext.autodoc',  # Core library for html generation from docstrings
              'sphinx.ext.autosummary',  # Create neat summary tables
              'sphinx.ext.todo',      # Support for todo items
              "myst_parser"
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosummary_generate_overwrite = True
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    "private-members": True,     # important for _foo and __foo
    "show-inheritance": True,
    'undoc-members': False,
    'exclude-members': '__weakref__'
}

# Configure autosummary to include class members by default
autosummary_context = {
    'skipmembers': set(),
}
templates_path = ['_templates']
bibtex_bibfiles = []
exclude_patterns = []
source_suffix = [".rst", ".md"]
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.insert(0, repo_root)  # Source code dir relative to this file
sys.path.insert(1, os.path.join(repo_root, "src"))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_css_files = ['custom.css']

# Add build time to footer
html_last_updated_fmt = None
html_theme_options = {
    "navigation_depth": -1,
    "includehidden": True,
    'version_selector': True,
    'prev_next_buttons_location': 'bottom',
}
html_show_sphinx = False
html_show_sourcelink = False
# html_logo = "512px-LOGO-IMEC_black.svg.png"
# html_theme_options = {
#     'logo_only': False,
#     'display_version': False,
# }
# html_css_files = ['custom_svg.css']
numfig = False

latex_elements = {
    "printindex": "", #noindex
    "classoptions": ",oneside,openany", # not empty pages
}

latex_domain_indices = False
