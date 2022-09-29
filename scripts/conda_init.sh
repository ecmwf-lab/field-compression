#!/usr/bin/env bash

set -e

if ! command -v conda &> /dev/null
then
    echo ""
    echo "Conda could not be located on your system. Please follow installation"
    echo "instructions at https://docs.conda.io/en/latest/miniconda.html"
    echo "before running this script."
    echo ""
    exit
fi

# activate conda environment
eval "$(conda shell.bash hook)"

if conda env list | grep -q fcpy; then
    # if fcpy env already exists update it
    conda env update --file environment.yml --prune
else
    # create a new env
    conda env create -f environment.yml
fi

# Julia deps
conda activate fcpy
# install julia dependecies
julia -e 'import Pkg; Pkg.add("PyCall")'
# specify the project on startup to build required dependencies
julia --project=. -e 'import Pkg; Pkg.instantiate()'

echo "Downloading sample data..."
mkdir -p data && cd data
[ -e "cams_q_20191201_v3.nc" ] && echo "cams_q_20191201_v3.nc already exists. Skipping..." || \
    curl -O https://files.codeocean.com/files/verified/e78744f2-1827-4eeb-95b3-dd5e828e9a71_v1.0/data/cams_q_20191201_v3.nc
[ -e "ensemble.t.member1.step0.ll.nc" ] && echo "ensemble.t.member1.step0.ll.nc already exists. Skipping..." || \
    curl -O https://files.codeocean.com/files/verified/e78744f2-1827-4eeb-95b3-dd5e828e9a71_v1.0/data/ensemble.t.member1.step0.ll.nc

