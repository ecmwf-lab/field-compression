#!/usr/bin/env bash

set -e

if ! command -v conda &> /dev/null
then
    echo ""
    echo "I could not find Anaconda on your system. Please follow installation"
    echo "instructions at https://docs.conda.io/en/latest/miniconda.html"
    echo "or check it is available in your PATH before rerunning this script."
    echo ""
    exit
fi

eval "$(conda shell.bash hook)"
conda activate fcpy

# TODO: fix metview in this env...
# set SKIP_NB=1 to skip generating notebooks
# METVIEW_PYTHON_ONLY=1
sphinx-build -v -b html docs/ docs/_build/