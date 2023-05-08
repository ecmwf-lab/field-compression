# (C) Copyright 2022 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np
import xarray as xr


def compute_bitinformation_single(da: xr.DataArray) -> xr.DataArray:
    from .julia import BitInformation

    vals = da.values.flatten()
    bitinf = BitInformation.bitinformation(vals)
    bitinf_da = xr.DataArray(data=bitinf, dims=["bit"], name="bitinf")
    return bitinf_da


def compute_required_bits_for_bitinf(
    bitinf: xr.DataArray, information_content: float
) -> int:
    bitinf = bitinf.values
    if len(bitinf) == 64:
        sign_exponent_bits = 12
    elif len(bitinf) == 32:
        sign_exponent_bits = 9
    else:
        raise NotImplementedError("unsupported dtype")
    required_mantissa_bits = (
        np.argmax(np.cumsum(bitinf) / np.sum(bitinf) >= information_content)
        # - sign_exponent_bits
    )
    return required_mantissa_bits
