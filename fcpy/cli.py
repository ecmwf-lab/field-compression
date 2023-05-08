# (C) Copyright 2022 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from typing import List

import fcpy


def create_compressors(specs: List[str]) -> List[fcpy.Compressor]:
    out = []
    for spec in specs:
        names = spec.split(",")
        compressor = None
        if not names:
            raise ValueError("Empty compressor spec")
        for name in names:
            try:
                compressor = getattr(fcpy, name)(compressor)
            except AttributeError:
                raise ValueError(f"Unknown compressor: {name}")
        out.append(compressor)
    return out


def subset_dataset(ds, specs: List[str]):
    for spec in specs:
        if "=" not in spec:
            raise ValueError("Invalid subset spec: " + spec)
        dim, sel = spec.split("=")
        sel = sel.split("-")
        if len(sel) == 1:
            ds = ds.isel(**{dim: [int(sel[0])]})
        elif len(sel) == 2:
            ds = ds.sel(**{dim: slice(float(sel[0]), float(sel[1]))})
        else:
            raise ValueError("Invalid subset spec: " + spec)
    return ds


def create_table(data: dict, stat_names):
    all_dims = set()
    for ds in data.values():
        for dim in ds.dims:
            if ds[dim].size > 1:
                all_dims.add(dim)
    all_dims = list(all_dims)
    fieldnames = ["var_name"] + all_dims + stat_names

    rows = []

    for var_name in data:
        ds = data[var_name]
        ds.load()

        other_dims = []
        d_dims = {}
        for dim in ds.dims:
            if ds[dim].size > 1:
                other_dims.append(dim)
                d_dims[dim] = ds[dim].values

        # Create a mesh with all the combinrations that we want to iterate through
        mesh = [val for val in product(*d_dims.values())]
        # Create a mapping dim keys and values so we can use in the selection
        dims_mapping = []
        for m in mesh:
            d_tmp = {}
            for i, dim in enumerate(other_dims):
                d_tmp[dim] = m[i]
            dims_mapping.append(d_tmp)

        for sel in dims_mapping:
            ds_sel = ds.sel(sel)
            entry = {dim: ds_sel[dim].item() for dim in other_dims}
            entry["var_name"] = var_name
            for stat_name in stat_names:
                entry[stat_name] = ds_sel[stat_name].item()
            rows.append(entry)

    return fieldnames, rows


def save_csv(fieldnames, rows, path):
    with open(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Dataset (.nc or .grib) or MARS request file (.json)",
    )
    parser.add_argument("--output", type=Path, default=Path("."), help="Output folder")
    parser.add_argument(
        "--dtype", choices=["float32"], help="Convert data to different type"
    )
    parser.add_argument(
        "--compressors",
        action="extend",
        nargs="+",
        default=["Round", "LinQuantization"],
        help="Example: --compressors Float,LinQuantization Log,Round",
    )
    parser.add_argument(
        "--vars", action="extend", nargs="+", help="Variables to process, otherwise all"
    )
    parser.add_argument(
        "--subset",
        action="extend",
        nargs="+",
        help="Variables to subset, e.g. --subset level=0-5",
    )

    args = parser.parse_args()

    compressors = create_compressors(args.compressors)
    if not compressors:
        raise ValueError("No compressors specified")

    if args.input.endswith(".json"):
        with open(args.input) as f:
            request = json.load(f)["mars"]
        ds = fcpy.fetch_mars_dataset(request)
    else:
        ds = fcpy.open_dataset(args.input)

    if args.vars:
        ds = ds[args.vars]

    if args.subset:
        ds = subset_dataset(ds, args.subset)

    if args.dtype:
        ds = ds.astype(args.dtype)

    data = fcpy.sigmas_iterate_ds(ds, compressors=compressors)

    fieldnames, rows = create_table(data, ["bits", "sigmas"])
    save_csv(fieldnames, rows, args.output / "bits.csv")

    fieldnames_bitinf, rows_bitinf = create_table(
        {name: ds.isel(compressor=0) for name, ds in data.items()}, ["bitinf"]
    )
    save_csv(fieldnames_bitinf, rows_bitinf, args.output / "bits_kloewer.csv")
