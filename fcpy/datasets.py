# (C) Copyright 2022 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import glob
from typing import Optional

import climetlab as cml
import xarray as xr

DATASET_ID = "climetlab-fields-compression"

PARAMETERS = {
    "atmospheric-model": {
        "ml": {
            "anoffset": 9,
            "class": "rd",
            "date": "2020-07-21",
            "expver": "hplp",
            "levelist": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                87,
                88,
                89,
                90,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                99,
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
                115,
                116,
                117,
                118,
                119,
                120,
                121,
                122,
                123,
                124,
                125,
                126,
                127,
                128,
                129,
                130,
                131,
                132,
                133,
                134,
                135,
                136,
                137,
            ],
            "levtype": "ml",
            "param": [
                "cc",
                "ciwc",
                "clwc",
                "crwc",
                "cswc",
                "o3",
                "q",
                # unsupported:
                # "lnsp",
                # "d",
                # "t",
                # "vo",
                # "w",
                # "z",
                # "u",
                # "v"
            ],
            "step": [
                0,
                12,
                24,
                36,
                48,
                60,
                72,
                84,
                96,
                108,
                120,
                132,
                144,
                156,
                168,
                180,
                192,
                204,
                216,
                228,
                240,
            ],
            "stream": "lwda",
            "time": "00:00:00",
            "type": "fc",
        },
        "sfc": {
            "anoffset": 9,
            "class": "rd",
            "date": "2020-07-21",
            "expver": "hplp",
            "levtype": "sfc",
            "param": [
                "10fg",
                "10u",
                "10v",
                "2d",
                "2t",
                "asn",
                "bld",
                "blh",
                "cape",
                "chnk",
                "ci",
                "cin",
                "cp",
                "crr",
                "csfr",
                "dsrp",
                "e",
                "es",
                "ewss",
                "fal",
                "flsr",
                "fsr",
                "fzra",
                "gwd",
                "hcc",
                "i10fg",
                "ie",
                "iews",
                # Ambiguous : ilspf could be INSTANTANEOUS SURFACE SENSIBLE HEAT FLUX or INSTANTANEOUS 10 METRE WIND GUST
                # "ilspf",
                "inss",
                "ishf",
                "istl1",
                "istl2",
                "istl3",
                "istl4",
                "kx",
                "lcc",
                "lgws",
                "lsm",
                "lsp",
                "lspf",
                "lsrr",
                "lssfr",
                "mcc",
                "mgws",
                "mn2t",
                "msl",
                "mx2t",
                "nsss",
                "ocu",
                "ocv",
                "par",
                "pev",
                "ptype",
                "ro",
                "rsn",
                "sd",
                "sf",
                "skt",
                "slhf",
                "smlt",
                "src",
                "sro",
                "sshf",
                "ssr",
                "ssrc",
                "ssrd",
                "ssro",
                "sst",
                "stl1",
                "stl2",
                "stl3",
                "stl4",
                "str",
                "strc",
                "strd",
                "sund",
                "swvl1",
                "swvl2",
                "swvl3",
                "swvl4",
                "tcc",
                "tciw",
                "tclw",
                "tco3",
                "tcrw",
                "tcsw",
                "tcw",
                "tcwv",
                "tisr",
                "totalx",
                "tp",
                "tsn",
                "tsr",
                "tsrc",
                "ttr",
                "ttrc",
                "uvb",
                "vimd",
                # Ambiguous : vis could be VERTICAL INTEGRAL OF NORTHWARD OZONE FLUX or VERTICALLY INTEGRATED MOISTURE DIVERGENCE
                # "vis",
                "z",
            ],
            "step": [
                0,
                12,
                24,
                36,
                48,
                60,
                72,
                84,
                96,
                108,
                120,
                132,
                144,
                156,
                168,
                180,
                192,
                204,
                216,
                228,
                240,
            ],
            "stream": "lwda",
            "time": "00:00:00",
            "type": "fc",
        },
    },
}

import climetlab_fields_compression.main


def no_validation(data, model, levtype, levels, param, step):
    pass


climetlab_fields_compression.main.validate_mars_request = no_validation


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def load_dataset(
    model: str,
    var_names: Optional[list] = None,
    levels: Optional[list] = None,
    steps: Optional[list] = None,
) -> xr.Dataset:
    """Load one of the pre-defined datasets via MARS.

    Args:
        model (str): The model to use, currently always "atmospheric-model".
        var_names (list, optional): Names of variables to load. Defaults to all.
        levels (list, optional): Levels to load. Defaults to all.
        steps (list, optional): Time steps to load. Defaults to all.

    Returns:
        xr.Dataset: The loaded dataset.
    """

    level_types = list(PARAMETERS[model].keys())

    var_names_all = {
        level_type: PARAMETERS[model][level_type]["param"] for level_type in level_types
    }

    var_names_all_flat = flatten(var_names_all.values())

    assert len(set(var_names_all)) == len(var_names_all)

    # Set defaults
    if not var_names:
        var_names = var_names_all_flat
    elif not all(x in var_names_all_flat for x in var_names):
        raise ValueError("Unknown variable")
    assert var_names is not None
    if not levels:
        levels = [1, 2]
    if not steps:
        steps = [0]

    datasets = []
    paths = []
    for level_type in level_types:
        if not any(x in var_names_all[level_type] for x in var_names):
            continue
        if "levelist" not in PARAMETERS[model][level_type]:
            levels_to_load = None
        else:
            levels_to_load = levels

        tmp_var_names = list(
            set(var_names_all[level_type]).intersection(set(var_names))
        )
        dataset = cml.load_dataset(
            DATASET_ID,
            model=model,
            levtype=level_type,
            param=tmp_var_names,
            levels=levels_to_load,
            step=steps,
        )
        paths.append(dataset[0].path)
        datasets.append(dataset.to_xarray())

    out = xr.merge(datasets)
    # Store path for regridding, see utils.py.
    out.attrs["path"] = paths[0]
    return out


def open_dataset(filepath) -> xr.Dataset:
    """Open a dataset from one or more local files.

    Args:
        filepath (str): Either .nc or .grib, may contain wildcards.

    Returns:
        xr.Dataset: The loaded dataset.
    """
    if filepath.endswith(".nc"):
        return xr.open_mfdataset(filepath)
    else:
        ds = cml.load_source("file", filepath).to_xarray(
            xarray_open_dataset_kwargs=dict(cache=False)
        )
        path = glob.glob(filepath)[0]
        ds.attrs["path"] = path
        return ds
