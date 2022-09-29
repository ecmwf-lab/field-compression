import numpy as np
import xarray as xr

from .julia import BitInformation


def compute_bitinformation_single(da: xr.DataArray) -> xr.DataArray:
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
