#!/usr/bin/env python

"""@package docstring
File: analysis_funcs.py
Author: Adam Lamson
Email: adam.r.lamson@gmail.com
Description:

"""

import re
import yaml
from pathlib import Path
import h5py
from copy import deepcopy

# Data manipulation
import numpy as np
import torch


# Data analysis helpers
def get_sorted_time_grps(h5_data):
    return sorted(
        [grp for grp in h5_data if grp.startswith("t")],
        key=lambda x: int(re.findall(r"\d+", x)[0]),
    )


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (
        (vec1 / np.linalg.norm(vec1)).reshape(3),
        (vec2 / np.linalg.norm(vec2)).reshape(3),
    )
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def extract_float_from_path(path):
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(path))
    return float(match.group()) if match else float("inf")


def get_cent_pos(data_dir, h5_data):
    """Get the center of mass of the tubulin in the simulation
    :param h5_data: h5py object
    :param time_grp: string, name of the group to extract data from
    :return: center of mass of the tubulin in the simulation
    """
    # Get centrosome positions
    params = yaml.safe_load(h5_data.attrs["params"])
    with (data_dir / params["centrosome_file"]).open("r") as f:
        centrosomes = np.loadtxt(f, delimiter=",")

    # Get vector between centrosomes and rotation matrix
    centrosome_vec = centrosomes[1, 1:4] - centrosomes[0, 1:4]
    centrosome_vec /= np.linalg.norm(centrosome_vec)
    rot_matrix = rotation_matrix_from_vectors(centrosome_vec, np.array([1, 0, 0]))
    cent0_pos = rot_matrix @ centrosomes[0, 1:4]
    cent1_pos = rot_matrix @ centrosomes[1, 1:4]
    return cent0_pos, cent1_pos


def calc_rad_tubulin_intensity(filaments, center, bin_width=0.025, max_dist=None):
    r_dist_minus_ends = torch.linalg.norm(filaments[:, 1:4] - center, axis=1)
    plus_ends = filaments[:, 1:4] + torch.einsum(
        "ij,i->ij", filaments[:, 4:7], filaments[:, 7]
    )
    r_dist_plus_ends = torch.linalg.norm(plus_ends - center, axis=1)

    if not max_dist:
        max_dist = max(r_dist_minus_ends.max(), r_dist_plus_ends.max())

    r_bins = torch.arange(0, max_dist + bin_width, bin_width)

    n_fils_in_bins_arr = torch.zeros_like(r_bins)

    for i, r in enumerate(r_bins):
        # Create conditions
        spanning_cond1 = (r_dist_minus_ends <= r) & (
            r_dist_plus_ends >= (r + bin_width)
        )
        spanning_cond2 = (r_dist_minus_ends >= (r + bin_width)) & (
            r_dist_plus_ends <= (r + bin_width)
        )
        minus_inside_cond = (r_dist_minus_ends > r) & (
            r_dist_minus_ends < (r + bin_width)
        )
        plus_inside_cond = (r_dist_plus_ends > r) & (r_dist_plus_ends < (r + bin_width))

        combined_cond = (
            spanning_cond1 | spanning_cond2 | minus_inside_cond | plus_inside_cond
        )
        n_fils_in_bins_arr[i] = combined_cond.sum()

    return r_bins, n_fils_in_bins_arr


# Graphing helpers
def plot_confidence_interval(ax, x, mean, std, n, color, alpha=0.3):
    z_score = 1.96  # Z-score for 95% confidence
    margin_of_error = z_score * (std / np.sqrt(n))
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    _ = ax.fill_between(x, lower_bound, upper_bound, color=color, alpha=alpha)
