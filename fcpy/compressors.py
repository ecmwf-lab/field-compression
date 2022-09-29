# (C) Copyright 2022 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple

import numpy as np

from .julia import BitInformation


class Compressor(metaclass=ABCMeta):
    """Abstract base class for compressors.

    Args:
        inner_compressor (Compressor, optional): Inner compressor to use, if any.
        bits (int, optional): Bits to use for compression. If None, must be given
            when calling compress().
    """

    def __init__(
        self,
        inner_compressor: Optional["Compressor"] = None,
        bits: Optional[int] = None,
    ):
        if inner_compressor is not None:
            assert isinstance(inner_compressor, Compressor)
        self.inner_compressor = inner_compressor
        self.bits = bits

    @property
    def name(self) -> str:
        """Name of the compressor."""
        s = type(self).__name__
        if self.inner_compressor:
            s = f"{s}({self.inner_compressor.name})"
        return s

    def compress(
        self, arr: np.ndarray, bits: Optional[int] = None
    ) -> Tuple[np.ndarray, list[dict]]:
        """Compress the given array.

        Args:
            arr (np.ndarray): Data to compress.
            bits (int, optional): Bits to use in compression.
                Defaults to bits passed in the constructor.

        Returns:
            Tuple[np.ndarray, list[dict]]:
                The compressed data with metadata of this and all
                inner compressors required for decompression.
        """
        bits = bits or self.bits
        assert bits is not None
        if self.inner_compressor:
            arr, params_stack = self.inner_compressor.compress(arr, bits=bits)
        else:
            params_stack = []
        c, params = self.do_compress(arr, bits)
        params_stack.insert(0, params)
        return c, params_stack

    def decompress(
        self, compressed_data: np.ndarray, params_stack: list[dict]
    ) -> np.ndarray:
        """Decompress the given data using the given compression metadata.

        Args:
            compressed_data (np.ndarray): The compressed data.
            params_stack (list[dict]): The compression metadata of this
                and all inner compressors.

        Returns:
            np.ndarray: The decompressed data.
        """
        params = params_stack.pop(0)
        d = self.do_decompress(compressed_data, params)
        if self.inner_compressor:
            d = self.inner_compressor.decompress(d, params_stack)
        return d

    @abstractmethod
    def do_compress(self, arr: np.ndarray, bits: int) -> Tuple[np.ndarray, dict]:
        """Method to be implemented in compressor subclasses.
        Called by the base class during compress().

        Args:
            arr (np.ndarray): Data to compress.
            bits (int): Bits to use for compression.

        Returns:
            Tuple[np.ndarray, dict]: Compressed data with metadata required for decompression.
        """
        raise NotImplementedError

    @abstractmethod
    def do_decompress(self, compressed_data: np.ndarray, params: dict) -> np.ndarray:
        """Method to be implemented in compressor subclasses.
        Called by the base class during decompress().

        Args:
            compressed_data (np.ndarray): The compressed data.
            params (dict): The compression metadata returned by do_compress().

        Returns:
            np.ndarray: The decompressed data.
        """
        raise NotImplementedError


class LinQuantization(Compressor):
    """Linear quantization compressor."""

    def do_compress(self, arr: np.ndarray, bits: int) -> Tuple[np.ndarray, dict]:
        minimum = np.amin(arr)
        maximum = np.amax(arr)
        arr_compressed = np.round(
            (arr - minimum) / (maximum - minimum) * (2**bits - 1)
        )
        return arr_compressed, {"bits": bits, "minimum": minimum, "maximum": maximum}

    def do_decompress(self, compressed_data: np.ndarray, params: dict) -> np.ndarray:
        bits = params["bits"]
        minimum = params["minimum"]
        maximum = params["maximum"]
        arr_decompressed = (
            np.array(compressed_data) / (2**bits - 1) * (maximum - minimum) + minimum
        )
        return arr_decompressed


class Round(Compressor):
    """Rounding compressor."""

    def get_used_sign_and_exponent_bits(self, arr: np.ndarray) -> int:
        arr = np.random.uniform(np.nanmin(arr), np.nanmax(arr), 1000).astype(arr.dtype)

        if arr.dtype == np.float32:
            dtype_int = np.uint32
            width = 32
            sign_and_exponent_bits = 9
        elif arr.dtype == np.float64:
            dtype_int = np.uint64
            width = 64
            sign_and_exponent_bits = 12
        else:
            raise RuntimeError("unsupported dtype")

        def to_bits(a):
            l = []
            for i in a:
                l.append(
                    np.array(
                        list(np.binary_repr(int(i.view(dtype=dtype_int)), width=width)),
                        dtype=dtype_int,
                    )
                )
            l = np.vstack(l)

            l2 = []
            for i in range(l.shape[1]):
                l2.append(l[:, i])

            return np.array(l2)

        bits_arr = to_bits(arr)

        used_sign_and_exponent_bits = 0
        for col in range(sign_and_exponent_bits):
            if all(bits_arr[col, :] == 0) or all(bits_arr[col, :] == 1):
                continue
            used_sign_and_exponent_bits += 1
        return used_sign_and_exponent_bits

    def do_compress(self, arr: np.ndarray, bits: int) -> Tuple[np.ndarray, dict]:
        used_sign_and_exponent_bits = self.get_used_sign_and_exponent_bits(arr)
        mantissa_bits = bits - used_sign_and_exponent_bits
        if mantissa_bits < 1:
            raise RuntimeError("Round: mantissa bits < 1, use higher bits value")
        return BitInformation.round(arr, mantissa_bits), {}

    def do_decompress(self, compressed_data: np.ndarray, params: dict) -> np.ndarray:
        return compressed_data


class Float(Compressor):
    """IEEE Floating-Point compressor."""

    DTYPE = {16: np.float16, 32: np.float32, 64: np.float64}

    def do_compress(self, arr: np.ndarray, bits: int) -> Tuple[np.ndarray, dict]:
        dtype = self.DTYPE[bits]
        return arr.astype(dtype, copy=False), {}

    def do_decompress(self, compressed_data: np.ndarray, params: dict) -> np.ndarray:
        return compressed_data


class Log(Compressor):
    """Log/Exp compressor, typically used for pre-/postprocessing."""

    def do_compress(self, arr: np.ndarray, bits: int) -> Tuple[np.ndarray, dict]:
        return np.log(arr), {}

    def do_decompress(self, compressed_data: np.ndarray, params: dict) -> np.ndarray:
        return np.exp(compressed_data)
