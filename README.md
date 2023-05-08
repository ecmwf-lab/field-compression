# Field Compression Laboratory <!-- no toc -->

[![CI](https://github.com/ecmwf-lab/field-compression/actions/workflows/ci.yml/badge.svg)](https://github.com/ecmwf-lab/field-compression/actions/workflows/ci.yml)
[![Available on pypi](https://img.shields.io/badge/Docs-https%3A%2F%2Fecmwf--lab.github.io%2Ffield--compression%2F-blue.svg)](https://ecmwf-lab.github.io/field-compression/)
[![Formatted with black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

## Contents  <!-- no toc -->
- [Field Compression Laboratory ](#field-compression-laboratory-)
  - [Contents  ](#contents--)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Set up](#set-up)
  - [How to use](#how-to-use)
  - [Example notebooks](#example-notebooks)
  - [Command-line interface](#command-line-interface)
  - [How to contribute](#how-to-contribute)
  - [Development notes](#development-notes)
  - [Copyright and license](#copyright-and-license)
  - [References](#references)

## Overview

The Field Compression Laboratory aims to evaluate the impact of lossy compression on the accuracy of meteorological quantities used in numerical weather prediction. The current framework includes a Python library (fcpy) and example notebooks. Currently, we support latitude/longitude and Gaussian gridded data in netCDF and GRIB formats.

***NOTE: fcpy is in alpha stage and being heavily refactored in several areas. Packaging, distribution, execution speed, and visual appearance are being reworked and will be finalised at a later stage. The current notebooks are not final and will help understand user needs iteratively.***


## Prerequisites
- Linux or macOS
- [Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links)


## Set up

To set-up or update the environment with the required dependencies and download sample data used in the examples run the following command from the command-line interface:

```sh
scripts/conda_init.sh
```


## How to use

Below is a minimal example of how to use fcpy to compare the effects of lossy compression on the relative error of specific humidity *q*. To set up an experiment in fcpy you need to load a GRIB or NetCDF dataset and create an fcpy suite defining a baseline, a list of compressors, and the type of metrics. If you wish to plot the data you can use the helper methods or functions provided or use your own.

```py
# There is a 30-second wait
# because of how we import julia packages
import matplotlib.pyplot as plt

import fcpy

# Loads data as an xarray Dataset
ds = fcpy.open_dataset("data/cams_q_20191201_v3.nc")
# Only select specific humidity q
ds = ds[["q"]]

# Define the suite. Here instead of telling fcpy how many
# bits to iterate through, we let it figure out based
# on Klöwer et al. (2021)'s bit-information metric.
suite = fcpy.Suite(
    ds=ds,
    baseline=fcpy.Float(bits=32),
    compressors=[
        fcpy.Round(),
        fcpy.Log(fcpy.LinQuantization()),  # <- nested compressor
    ],
    metrics=[fcpy.RelativeError, fcpy.AbsoluteError],
    bits=None,  # <- computes number of bits using Klöwer et al. (2021)'s bit-information
)

# Plot the maximum relative error per bit and compressor combination
suite.lineplot(fcpy.RelativeError, reduction="max")
plt.savefig("sample.png", dpi=300)
```

For options please refer to the [API documentation]().


## Example notebooks

The easiest to start is by running the Jupyter Notebooks under `notebooks/` with the following command:

```sh
scripts/conda_run_notebooks.sh
```

There you will see two example notebooks named `examples-interactive` and `examples-programmatic`. The former shows how to call interactive plots and the latter programmatically.


## Command-line interface

The fcpy command-line interface offers an easy way to determine the number of bits required per variable and dimensions in a CSV table.

```sh
fcpy --input data/cams_q_20191201_v3.nc --vars q --subset lev=0-10
```

This will create the following CSV output table:

```
var_name,lev,compressor,bits,sigmas
q,1.0,Round,14.0,0.32247692346572876
q,2.0,Round,14.0,0.4317324161529541
q,3.0,Round,14.0,0.493638277053833
q,4.0,Round,15.0,0.5703839063644409
q,5.0,Round,15.0,0.6555898189544678
q,6.0,Round,16.0,0.774020254611969
q,7.0,Round,17.0,0.8185981512069702
q,8.0,Round,18.0,0.8566567897796631
q,9.0,Round,19.0,0.8976887464523315
q,10.0,Round,17.0,0.9218334555625916
q,1.0,LinQuantization,13.0,0.9965649843215942
q,2.0,LinQuantization,13.0,0.9766294956207275
q,3.0,LinQuantization,12.0,0.966712474822998
q,4.0,LinQuantization,12.0,0.9503071904182434
q,5.0,LinQuantization,12.0,0.9498687982559204
q,6.0,LinQuantization,14.0,0.9693909883499146
q,7.0,LinQuantization,15.0,0.9825363159179688
q,8.0,LinQuantization,15.0,0.9840608239173889
q,9.0,LinQuantization,16.0,1.0056097507476807
q,10.0,LinQuantization,16.0,0.9886303544044495
```

For more info on how to use the tool, run `fcpy --help`:

```
usage: fcpy [-h] --input INPUT [--output OUTPUT] [--dtype {float32}] [--compressors COMPRESSORS [COMPRESSORS ...]] [--vars VARS [VARS ...]]
            [--subset SUBSET [SUBSET ...]]

options:
  -h, --help            show this help message and exit
  --input INPUT         Dataset (.nc or .grib) or MARS request file (.json)
  --output OUTPUT       Output folder
  --dtype {float32}     Convert data to different type
  --compressors COMPRESSORS [COMPRESSORS ...]
                        Example: --compressors Float,LinQuantization Log,Round
  --vars VARS [VARS ...]
                        Variables to process, otherwise all
  --subset SUBSET [SUBSET ...]
                        Variables to subset, e.g. --subset level=0-5
```

## How to contribute

See [CONTRIBUTING.md](CONTRIBUTING.md)


## Development notes

See [DEVELOP.md](DEVELOP.md)


## Copyright and license

Copyright 2022 ECMWF. Licensed under [Apache License 2.0](LICENSE.txt). In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of its status as an intergovernmental organisation nor does it submit to any jurisdiction.

## References

```
Klöwer, M., Razinger, M., Dominguez, J.J. et al. Compressing atmospheric data into its real information content. Nat Comput Sci 1, 713–724 (2021). https://doi.org/10.1038/s43588-021-00156-2
```
