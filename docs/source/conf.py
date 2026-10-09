
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys, os
import inspect
from pathlib import Path

from sphinx.util import logging
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path('../..', 'ptypy').resolve()))
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# Hack to solve import issues with mocked libraries like fabio etc.
# Save the original unwrap function
original_unwrap = inspect.unwrap

def safe_unwrap(func, *args, **kwargs):
    # If it's a Sphinx Mock object, prevent it from looping
    if type(func).__name__ == '_MockObject' or hasattr(func, '__sphinx_mock__'):
        return func
    try:
        return original_unwrap(func, *args, **kwargs)
    except ValueError as e:
        if "wrapper loop when unwrapping" in str(e):
            return func
        raise

# Overwrite the standard library inspect.unwrap with our safe version
inspect.unwrap = safe_unwrap
# End of hack

# Check environment variables to decide what to generate as part of the build
generate_legacy_tutorials = int(os.environ["PTYPY_BUILD_LEGACY_TUTORIALS"]) if "PTYPY_BUILD_LEGACY_TUTORIALS" in os.environ else 0
generate_param_tree = int(os.environ["PTYPY_BUILD_PARAM_TREE"]) if "PTYPY_BUILD_PARAM_TREE" in os.environ else 0
generate_userguide_images = int(os.environ["PTYPY_BUILD_USERGUIDE_IMAGES"]) if "PTYPY_BUILD_USERGUIDE_IMAGES" in os.environ else 0
generate_notebook_tutorials = int(os.environ["PTYPY_BUILD_NOTEBOOK_TUTORIALS"]) if "PTYPY_BUILD_NOTEBOOK_TUTORIALS" in os.environ else 0
path_to_notebooks = os.environ["PTYPY_PATH_TO_NOTEBOOKS"] if "PTYPY_PATH_TO_NOTEBOOKS" in os.environ else None

# Generate List of Parameters
if generate_param_tree:
    from _param_generator import generate_parameters_rst
    #generate_parameters_rst("ptycho", outfile="ptycho.rst", title="Root/Ptycho (p)")
    generate_parameters_rst("io", outfile="io.rst", title="Input/Output (p.io)")
    generate_parameters_rst("scans", outfile="scans.rst", title="List of Scans (p.scans)")
    generate_parameters_rst("scan", outfile="scan.rst", title="Scan Definition (p.scans.scan_00)")
    generate_parameters_rst("scandata", outfile="scandata.rst", title="Scan Data Definition (p.scans.scan_00.data)")
    generate_parameters_rst("engines", outfile="engines.rst", title="List of Engines (p.engines)")
    generate_parameters_rst("engine", outfile="engine.rst", title="Engine Definition (p.engines.engine_00)")

# Create folder generated in userguide
os.makedirs("./userguide/generated", exist_ok=True)

# Generate images for user guide
if generate_userguide_images:
    from _userguide_generator import create_test_image
    create_test_image(outdir="./userguide/generated/", outfile="test.png")
    from _userguide_generator import create_all_init_probe_figures
    create_all_init_probe_figures(outdir="./userguide/generated/")

# Generate legacy tutorials
if generate_legacy_tutorials:
    from _legacy_tutorial_generator import generate_legacy_tutorial_rst
    generate_legacy_tutorial_rst("minimal_script.py", outdir="userguide/generated/legacy/")
    generate_legacy_tutorial_rst("ptypyclasses.py", outdir="userguide/generated/legacy/")
    generate_legacy_tutorial_rst("simupod.py", outdir="userguide/generated/legacy/")
    generate_legacy_tutorial_rst("ownengine.py", outdir="userguide/generated/legacy/")
    generate_legacy_tutorial_rst("subclassptyscan.py", outdir="userguide/generated/legacy/")

    from _legacy_tutorial_generator import replace_rst_in_templates
    replace_rst_in_templates("./userguide/rst_templates/getting_started.tmp")
    replace_rst_in_templates("./userguide/rst_templates/concept.tmp")
    replace_rst_in_templates("./userguide/rst_templates/data_management.tmp")

# Convert notebooks
path_to_generated_notebooks = "./userguide/generated/notebooks"
if generate_notebook_tutorials and (path_to_notebooks is not None):
    if not os.path.lexists(path_to_generated_notebooks):
        os.symlink(path_to_notebooks, path_to_generated_notebooks)

# Extract doc version from env variable
version_match = os.environ.get("PTYPY_DOCS_VERSION")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

project = 'PtyPy'
copyright = '2024, Pierre Thibault, Bjoern Enders, Benedikt Daurer and others'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'myst_nb',
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence"
]

nb_execution_mode = 'off'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

rst_epilog = """
.. |ptypy| replace:: PtyPy
.. _ptypy: https://www.github.com/ptycho/ptypy
"""

autosummary_generate = True
autodoc_mock_imports = ["cupy", "pycuda", "reikna", "hdf5plugin", "bitshuffle", "fabio", "swmr_tools"]

todo_include_todos = True
numfig = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ["ptypy.css"]
html_logo = '_static/logo_100px.png'
html_favicon = '_static/ptypyicon.ico'
html_show_sourcelink = False
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
    "switcher": {
        "json_url": "https://ptycho.github.io/ptypy/switcher.json",
        "version_match": version_match,
    },
    "navbar_start": ["navbar-logo", "version-switcher"]
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
