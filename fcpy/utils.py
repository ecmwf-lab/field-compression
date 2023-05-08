# (C) Copyright 2022 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import tempfile
from typing import Optional

import climetlab as cml
import eccodes as ecc
import numpy as np
import xarray as xr


def regrid(
    source: xr.DataArray,
    path_to_template: str,
    dx: float = 1,
    dy: float = 1,
    new_labels: Optional[dict] = None,
) -> xr.DataArray:
    """Regrid a Gaussian grid field into a regular latitude longitude grid.

    Args:
        source (xr.DataArray): field in Gaussian grid
        path_to_template (str): path to GRIB in Gaussian grid
        dx (int, optional): longitude grid resolution in arc-degree. Defaults to 1.
        dy (int, optional): latitude grid resolution in arc-degree. Defaults to 1.
        new_labels (Optional[dict], optional): Name of regridded xr.DataArray. Defaults to None.

    Returns:
        xr.DataArray: Regular latitude longitude grid
    """

    try:
        import metview as mv
    except:
        raise ImportError("Metview could not be imported.")

    assert isinstance(source, xr.DataArray), "source must be of type xarray.DataArray"

    with open(path_to_template, "rb") as file_template:
        with tempfile.NamedTemporaryFile("wb") as file_filled:

            # Read the first message in the file
            msgid = ecc.codes_grib_new_from_file(file_template)

            # Get data/metadata
            paramId = ecc.codes_get(msgid, "paramId")

            # Set data/metadata
            # TODO: set missing value
            # MISSING_VALUE = 9998
            # ecc.codes_set(msgid, 'missingValue', MISSING_VALUE)

            # overwrite the paramId with an arbitrary experimental paramId
            # to avoid issues with interpolation in case of special paramId
            ecc.codes_set(msgid, "paramId", 80)

            ecc.codes_set_values(msgid, source.values.flatten())

            ecc.codes_write(msgid, file_filled)

            ecc.codes_release(msgid)

            # Regrid data
            target_grid = {"grid": [dx, dy], "interpolation": "nearest_neighbour"}

            data = mv.read(file_filled.name)
            with tempfile.TemporaryDirectory() as tmpdirname:
                path_to_regridded = f"{tmpdirname}/file_regridded.grib"

                _ = mv.regrid(target_grid, data=data, target=path_to_regridded)

                ds_regridded = cml.load_source("file", path_to_regridded).to_xarray()
                assert len(list(ds_regridded)) == 1
                da_regridded = ds_regridded[list(ds_regridded)[0]]

        da_regridded.attrs = {
            "long_name": source.attrs["long_name"],
            "units": source.attrs["units"],
        }
        da_regridded.name = source.name

    return da_regridded


STANDARD_NAME_LAT = "latitude"
STANDARD_NAME_LON = "longitude"


def get_standard_name_dims(ds: xr.Dataset) -> dict:
    """Return the standard_name of each dimention in the dataset

    Args:
        ds (xr.Dataset): _description_

    Returns:
        dict: _description_
    """
    return {
        c.standard_name: c.name
        for c in [ds[dim] for dim in ds.dims]
        if "standard_name" in c.attrs
    }


def compute_z_score(da: xr.DataArray) -> xr.DataArray:
    """Normalize array to [0,1]"""
    return (da - da.min()) / (da.max() - da.min())


def get_bits_params(da: xr.DataArray) -> dict:
    if da.dtype == np.float32:
        dtype_int = np.uint32
        width = 32
        sign_and_exponent_bits = 9
    elif da.dtype == np.float64:
        dtype_int = np.uint64
        width = 64
        sign_and_exponent_bits = 12
    else:
        raise RuntimeError("unsupported dtype")
    return dict(
        dtype_int=dtype_int, width=width, sign_and_exponent_bits=sign_and_exponent_bits
    )


def to_bits(da, bits_params):
    value_range = np.linspace(da.min(), da.max(), 10000).astype(da.dtype)
    l = []
    for i in value_range:
        l.append(
            np.array(
                list(
                    np.binary_repr(
                        int(i.view(dtype=bits_params["dtype_int"])),
                        width=bits_params["width"],
                    )
                ),
                dtype=bits_params["dtype_int"],
            )
        )
    l = np.vstack(l)
    l2 = []
    for i in range(l.shape[1]):
        l2.append(l[:, i])

    return np.array(l2)


def compute_min_bits(da, bits_params):

    bits_arr = to_bits(da, bits_params)
    used_sign_and_exponent_bits = 0
    for col in range(bits_params["sign_and_exponent_bits"]):
        if all(bits_arr[col, :] == 0) or all(bits_arr[col, :] == 1):
            continue
        used_sign_and_exponent_bits += 1
    return used_sign_and_exponent_bits
