import os
import shutil
import sys
from pathlib import Path

os.environ["SPHINX_BUILD"] = "1"

PROJ_DIR = Path.cwd().parent
sys.path.insert(0, str(PROJ_DIR.resolve()))


# Copy root dir files
LICENSE_PREAMBLE = (
    "# Copyright & License\n\n Copyright 2022 ECMWF. Licensed under Apache License 2.0 (see text below)."
    "In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue "
    "of its status as an intergovernmental organisation nor does it submit to any jurisdiction.\n\n"
)


def copy_content(f_read: Path, f_write: Path, rebuild=False) -> None:
    if rebuild:
        with open(f_read) as fr, open(f_write, "w") as fw:
            if f_write.stem == "license":
                header = "# Copyright & License\n\n"
                fw.write(LICENSE_PREAMBLE)
                fw.write("```")
            content = fr.readlines()
            fw.writelines(content)
            if f_write.stem == "license":
                fw.write("```")


copy_content(PROJ_DIR / "CONTRIBUTING.md", PROJ_DIR / "docs" / "contributing.md", True)
copy_content(PROJ_DIR / "DEVELOP.md", PROJ_DIR / "docs" / "develop.md", True)
copy_content(PROJ_DIR / "LICENSE.txt", PROJ_DIR / "docs" / "license.md", True)

# Copy info from README.md
def copy_overview(f_read: Path, f_write: Path, rebuild=False) -> None:
    if rebuild:
        with open(f_read) as fr, open(f_write, "w") as fw:
            parse = False
            content = fr.readlines()
            fw.write("# Overview\n")
            for line in content:
                if line.startswith("The Field Compression Laboratory"):
                    parse = True
                if line.startswith("## How to contribute"):
                    parse = False
                if parse:
                    fw.write(line)


copy_overview(PROJ_DIR / "README.md", PROJ_DIR / "docs" / "overview.md", True)

# Copy notebooks folder
shutil.copytree(
    PROJ_DIR / "notebooks", PROJ_DIR / "docs" / "_notebooks", dirs_exist_ok=True
)

# Copy data folder
shutil.copytree(PROJ_DIR / "data", PROJ_DIR / "docs" / "data", dirs_exist_ok=True)

# Copy samples
shutil.copytree(PROJ_DIR / "samples", PROJ_DIR / "docs" / "samples", dirs_exist_ok=True)


project = "fcpy"
copyright = "2022 ECMWF"
author = "ECMWF"
release = "0.2.0"

html_context = {
    "display_github": True,
    "github_user": "ecmwf-lab",
    "github_repo": "field-compression",
    "github_version": "main/docs/",
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "myst_parser",
]

bibtex_bibfiles = ["references.bib"]

# Autodoc settings
autodoc_typehints = "none"  # 'signature'
autodoc_inherit_docstrings = False

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = False

copybutton_prompt_text = ">>> "


nbsphinx_execute = "always"
# Some notebooks take longer than the default 30 s limit.
nbsphinx_timeout = 600

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
if os.environ.get("SKIP_NB") == "1":
    exclude_patterns.append("_notebooks")

html_theme = "sphinx_book_theme"

html_theme_options = {
    "collapse_navigation": False,
    "display_version": False,
    "logo_only": True,
}
