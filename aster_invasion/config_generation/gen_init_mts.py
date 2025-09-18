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
        prog="gen_init_mts.py", description="Parse arguments for gen_init_mts."
    )

    parser.add_argument(
        "filepath",
        type=Path,
        help="File name for operations of arguments. Either hdf5 file or yaml file.",
    )

    parser.add_argument(
        "-d",
        "--distribution",
        choices=["mono", "exp"],
        default="mono",
        help="Choose the distribution of MT lengths around centrosomes.",
    )

    parser.add_argument(
        "-s",
        "--spacing",
        choices=["random", "equal"],
        default="random",
        help="Placement of MTs on centrosomes. Only for 2D.",
    )

    parser.add_argument(
        "-L",
        "--length",
        type=float,
        nargs="*",
        default=[],
        help="Set the average length of MTs manually.",
    )

    parser.add_argument(
        "-M",
        "--mt_per_centrosome",
        type=int,
        nargs="*",
        default=[],
        help="Set the number of MTs per centrosome manually. All will be set to growing.",
    )

    parser.add_argument(
        "--hdf5",
        type=int,
        default=None,
        help="Generate initial MTs from a frame from an hdf5 file. Specify the frame number. Filepath must be an hdf5 file. (Not implemented yet)",
    )

    args = parser.parse_args()
    return args


def sample_unit_sphere(n_points, args):
    # Generate azimuthal angles (phi) between 0 and 2*pi
    # Generate random polar angles (theta) with the correct distribution
    if args.spacing == "equal":
        phi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        theta = np.arcos(1 - 2 * np.linspace(0, 1, n_points, endpoint=False))
    else:
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        theta = np.arccos(1 - 2 * np.random.uniform(0, 1, n_points))

    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Combine x, y, z coordinates into a single array
    points = np.vstack((x, y, z)).T

    return points


def sample_unit_circle(n_points, args):
    # Generate random azimuthal angles (phi) uniformly distributed between 0
    # and 2*pi
    if args.spacing == "equal":
        phi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    else:
        phi = np.random.uniform(0, 2 * np.pi, n_points)

    # Convert spherical coordinates to Cartesian coordinates
    x = np.cos(phi)
    y = np.sin(phi)
    z = np.zeros(n_points)

    # Combine x, y, z coordinates into a single array
    points = np.vstack((x, y, z)).T

    return points


def get_mt_lengths(n_mts, avg_length, args):
    if args.distribution == "exp":
        lengths = np.random.exponential(avg_length, n_mts)
    elif args.distribution == "mono":
        lengths = np.ones(n_mts) * avg_length

    return lengths


def main(params, args):
    # Calculate necessary parameters
    dimension = params["dimension"]
    np.random.seed(params["rng_seed"])

    # Load centrosomes
    cents = np.loadtxt(params["centrosome_file"], delimiter=",", ndmin=2)

    mts = np.zeros((0, 11))
    # Loop over centrosomes and generate MTs
    cur_mt_id = 0
    for i_cent, cent in enumerate(cents):
        if len(args.mt_per_centrosome) == 1:
            n_grow_mts = args.mt_per_centrosome[0]
            n_shrink_mts = 0
        elif len(args.mt_per_centrosome) > 1:
            n_grow_mts = args.mt_per_centrosome[i_cent]
            n_shrink_mts = 0
        else:
            n_grow_mts = int(params["nuc_rate"] / params["kc"])
            n_shrink_mts = int(n_grow_mts * (params["vg"] / params["vs"]))
        n_mts = n_grow_mts + n_shrink_mts

        if len(args.length) == 1:
            avg_mt_length = args.length[0]
        elif len(args.length) > 1:
            avg_mt_length = args.length[i_cent]
        else:
            avg_mt_length = params["vg"] / params["kc"]

        cent_rad = cent[4]

        # Growing MTs
        ids = np.arange(cur_mt_id, cur_mt_id + n_grow_mts)
        unit_points = (
            sample_unit_circle(n_grow_mts, args)
            if dimension == 2
            else sample_unit_sphere(n_grow_mts, args)
        )

        lengths = get_mt_lengths(n_grow_mts, avg_mt_length, args)

        grow_mts = np.hstack(
            (
                ids[:, None],  # ID
                cent[1:4] + cent_rad * unit_points,  # Minus end
                unit_points,  # Direction
                lengths[:, None],  # Length
                np.ones(n_grow_mts)[:, None],  # State = grow
                -np.ones(n_grow_mts)[:, None],  # Nucleator id = None
                cent[0] * np.ones(n_grow_mts)[:, None],  # Centrosome id
            )
        )
        cur_mt_id += n_grow_mts

        # Shrinking MTs
        ids = np.arange(cur_mt_id, cur_mt_id + n_shrink_mts)
        unit_points = (
            sample_unit_circle(n_shrink_mts, args)
            if dimension == 2
            else sample_unit_sphere(n_shrink_mts, args)
        )
        lengths = get_mt_lengths(n_shrink_mts, avg_mt_length, args)
        shrink_mts = np.hstack(
            (
                ids[:, None],  # ID
                cent[1:4] + cent_rad * unit_points,  # Minus end
                unit_points,  # Direction
                lengths[:, None],  # Length
                -np.ones(n_shrink_mts)[:, None],  # State = shrink
                -np.ones(n_shrink_mts)[:, None],  # Nucleator id = None
                cent[0] * np.ones(n_shrink_mts)[:, None],  # Centrosome id
            )
        )
        cur_mt_id += n_shrink_mts

        mts = np.vstack((mts, grow_mts, shrink_mts))

    print(f"Generated {mts.shape[0]} MTs.")
    print(f" Number of centrosomes: {cents.shape[0]}")
    print(
        f" Number of MTs (grow): {np.sum(mts[:, 8] == 1)} ({n_grow_mts} per centrosome)"
    )
    print(
        f" Number of MTs (shrink): {np.sum(mts[:, 8] == -1)} ({n_shrink_mts} per centrosome)"
    )
    print(f"Average MT length: {np.mean(mts[:, 7])}")
    if n_grow_mts > 0:
        print(f"Average MT length (grow): {np.mean(mts[mts[:, 8] == 1, 7])}")
    if n_shrink_mts > 0:
        print(f"Average MT length (shrink): {np.mean(mts[mts[:, 8] == -1, 7])}")

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
    if args.hdf5:
        print("HDF5 functionality not implemented yet.")
        sys.exit()

    param_file = args.filepath
    with open(param_file) as f:
        params = yaml.safe_load(f)

    main(params, args)
