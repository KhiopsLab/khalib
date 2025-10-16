project = "khalib"
author = "Felipe Olmos"
release = "0.1"

extensions = [
    "myst_nb",  # activates myst_parser as well
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx_copybutton",
    "sphinx_paramlinks",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"<h6><center>{project} {release}</center></h6>"

autosummary_generate = True
autosummary_ignore_module_all = False

default_role = "obj"

# Enable this to debug notebooks
# nb_execution_allow_errors=True
nb_execution_show_tb: True


## Numpydoc extension config
numpydoc_show_class_members = False

## Autodoc extension config
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "private-members": False,
    "show-inheritance": True,
    "special-members": False,
}

## Intersphinx extension config
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}
