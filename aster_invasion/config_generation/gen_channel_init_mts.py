#!/usr/bin/env python

"""@package docstring
File: gen_init_mts.py
Author: Adam Lamson
Email: adam.r.lamson@gmail.com
Description:

"""

import sys
import numpy as np
import h5py
import yaml

import argparse
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="gen_channel_init_mts.py", description="Parse arguments for gen_init_mts."
    )

    parser.add_argument(
        "filepath",
        type=Path,
        help="File name for operations of arguments. Either hdf5 file or yaml file.",
    )

    parser.add_argument(
        "-L",
        "--length",
        type=float,
        default=-1,
        help="Set the length of MTs manually.",
    )

    parser.add_argument(
        "-M",
        "--mt_number",
        type=int,
        default=-1,
        help="Set the number of MTs.",
    )

    args = parser.parse_args()
    return args


def main(params, args):
    # Calculate necessary parameters
    n_mts = args.mt_number
    length = args.length
    # dimension = params["dimension"]
    hLy = 0.5 * params["Ly"]
    hLz = 0.5 * params["Lz"]

    ids = np.arange(n_mts)
    # Make an array of xy points that create a lattice around the origin
    n = int(np.sqrt(n_mts))
    y = np.linspace(-hLy, hLy, n + 2)
    z = np.linspace(-hLz, hLz, n + 2)
    y, z = np.meshgrid(y[1:-1], z[1:-1])
    y = y.flatten()
    z = z.flatten()
    x = np.zeros_like(z)
    pos = np.vstack((x, y, z)).T
    dir = np.zeros_like(pos)
    dir[:, 0] = 1
    lengths = np.ones_like(x) * length

    mts = np.hstack(
        (
            ids[:, None],  # ID
            pos,  # Minus end
            dir,  # Direction
            lengths[:, None],  # Length
            np.ones_like(x)[:, None],  # State = grow
            -np.ones_like(x)[:, None],  # Nucleator id = None
            np.zeros_like(x)[:, None],  # Centrosome id
        )
    )

    print(f"Generated {mts.shape[0]} MTs.")
    print(mts)

    np.savetxt(
        "mt_init.csv",
        mts,
        delimiter=",",
        fmt=[
            "%d",
            "%.5f",
            "%.5f",
            "%.5f",
            "%.5f",
            "%.5f",
            "%.5f",
            "%.5f",
            "%d",
            "%d",
            "%d",
        ],
    )


if __name__ == "__main__":
    # param_file = sys.argv[1]
    args = parse_arguments()

    param_file = args.filepath
    with open(param_file) as f:
        params = yaml.safe_load(f)

    main(params, args)
