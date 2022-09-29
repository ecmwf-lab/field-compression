# This is the readme example

# How to run:
# $ conda activate fcpy
# $ python samples/simple.py

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
