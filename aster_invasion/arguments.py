#!/usr/bin/env python

import argparse
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="aster_main.py", description="Parse arguments for aster_main"
    )

    parser.add_argument(
        "filepath",
        type=Path,
        help="File name for operations of arguments.",
    )

    parser.add_argument(
        "-h2v",
        "--hdf5_to_vtk",
        action="store_true",
        help="Call hdf5 to vtk conversion. Uses the filepath variable.",
    )

    args = parser.parse_args()
    return args
