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

download_data () {
    fname="$1"
    if [ $fname == mars.grib ]; then
        url="https://raw.githubusercontent.com/ecmwf-lab/field-compression/data/mars.grib"
    else
        url="https://files.codeocean.com/files/verified/e78744f2-1827-4eeb-95b3-dd5e828e9a71_v1.0/data/${fname}"
    fi

    [ -e "${fname}" ] && echo "${fname} already exists. Skipping..." || \
    curl -O $url
}

for fname in mars.grib cams_ch4_20191201_v3.nc cams_co_20191201_v3.nc cams_co2_20191201_v3.nc \
             cams_go3_20191201_v3.nc cams_no2_20191201_v3.nc cams_q_20191201_v3.nc \
             cams_so2_20191201_v3.nc ensemble.t.member1.step0.ll.nc
do
    download_data $fname
done
