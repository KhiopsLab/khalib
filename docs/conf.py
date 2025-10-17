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
default_role = "obj"

# Extensions configs

## autosummary
autosummary_generate = True
autosummary_ignore_module_all = False

## autodoc
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "private-members": False,
    "show-inheritance": True,
    "special-members": False,
}


## myst_nb
nb_execution_allow_errors = False  # Enable this to debug notebooks
nb_execution_show_tb: True


## numpydoc
numpydoc_show_class_members = False


## intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}
