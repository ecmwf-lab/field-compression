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
jupyter notebook notebooks/
