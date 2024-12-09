# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

sys.path.insert(0, str(Path('../..', 'ptypy').resolve()))


project = 'PtyPy'
copyright = '2024, Pierre Thibault, Bjoern Enders, Benedikt Daurer and others'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    # 'sphinx.ext.linkcode',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

rst_epilog = """
.. |ptypy| replace:: PtyPy
.. _ptypy: https://www.github.com/ptycho/ptypy
"""

autosummary_generate = True
autodoc_mock_imports = ["cupy", "pycuda", "reikna", "hdf5plugin", "bitshuffle", "fabio", "swmr_tools"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ["ptypy.css"]
html_logo = '_static/logo_100px.png'
html_favicon = '_static/ptypyicon.ico'
html_sidebars = {
    'overview': []
    }

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ptycho/ptypy",
            "icon": "fab fa-github-square",
        },
        {
            "name": "ptypy.org",
            "url": "https://ptypy.org/",
            "icon": "fa-solid fa-link ",
        },
    ],
    "announcement": "https://daurer.github.io/ptypy-new-docs/docs/_static/banner.html",
}


# -- Custom functions ----------------------------------------------------

def truncate_docstring(app, what, name, obj, options, lines):
    """
    Remove the Default parameter entries.
    """
    if not hasattr(obj, 'DEFAULT'):
        return
    if any(l.strip().startswith('Defaults:') for l in lines):
        while True:
            if lines.pop(-1).strip().startswith('Defaults:'):
                break


def remove_mod_docstring(app, what, name, obj, options, lines):
    from ptypy import utils as u
    from ptypy import defaults_tree
    u.verbose.report.headernewline='\n\n'
    searchstr = ':py:data:'
    
    def get_refs(dct, pd, depth=2, indent=''):
        if depth < 0:
            return
        
        for k, value in dct.items():
            ref = ', see :py:data:`~%s`' % pd.children[k].entry_point if k in pd.children else ''
            if hasattr(value, 'items'):
                v = str(value.__class__.__name__)
            elif str(value) == value:
                v = '"%s"' % value
            else:
                v = str(value)
                
            lines.append(indent + '* *' + k + '* = ``' + v + '``' + ref)
            
            if hasattr(value, 'items'):
                lines.append("")
                get_refs(value, pd.children[k], depth=depth-1, indent=indent+'  ')
                lines.append("")

    if isinstance(obj, u.Param) or isinstance(obj, dict):
        pd = None
        
        for l in lines:
            start = l.find(searchstr)
            if start > -1:
                newstr = l[start:]
                newstr = newstr.split('`')[1]
                newstr = newstr.replace('~', '')
                pd = defaults_tree.get(newstr)
                break
                
        if pd is not None:
            get_refs(obj, pd, depth=2, indent='')

        
def setup(app):
    print("Custom setup")
    app.connect('autodoc-process-docstring', remove_mod_docstring)
    app.connect('autodoc-process-docstring', truncate_docstring)
    pass
