# Field Compression Laboratory <!-- no toc -->

[![CI]]
[![Docs]]
[![Formatted with black]]

## Contents  <!-- no toc -->
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Set up](#set-up)
- [How to use](#how-to-use)
- [Example notebooks](#example-notebooks)
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
