#!/usr/bin/env python

"""@package docstring
File: filament_funcs.py
Author: Adam Lamson
Email: alamson@flatironinstitute.org
Description:

Units: [F] = pN, [L] = um, [T] = s
"""

import torch
import sys
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64

import numba
import gc
from math import exp, acos

HALF_PI = 0.5 * torch.pi
float_type = numba.float64
TPB = 128  # threads per block
max_value = 3.402823e37


def check_for_sorted_and_uniqueness(fil_ids):
    """Check if filament ids are unique and sorted"""
    if not torch.all(fil_ids[:-1] < fil_ids[1:]):
        raise ValueError("Filament ids must be sorted and unique.")
    return


def change_filament_lengths(filaments, growing_idx, shrinking_idx, params):
    """
    Updates the lengths of filaments by growing or shrinking them according to specified indices and parameters.

    Parameters:
        filaments (torch.Tensor): A 2D tensor where each row represents a filament and the 8th column (index 7) stores the filament length.
        growing_idx (torch.Tensor): 1D tensor of indices indicating growing filaments.
        shrinking_idx (torch.Tensor): 1D tensor of indices indicating shrinking filaments.
        params (dict): Dictionary containing the following keys:
            - "vg" (float): Growth velocity to apply to growing filaments.
            - "vs" (float): Shrinkage velocity to apply to shrinking filaments.
            - "dt" (float): Time step for the update.

    Raises:
        ValueError: If any index appears in both growing_idx and shrinking_idx.
    """
    vg = params["vg"]
    vs = params["vs"]
    dt = params["dt"]
    # Check to make sure indices do not appear in both growing and shrinking
    if torch.any(torch.isin(growing_idx, shrinking_idx)):
        raise ValueError(
            "Indices for growing and shrinking filaments must not overlap."
        )

    # Grow filaments
    filaments[growing_idx, 7] += vg * dt

    # Shrink filaments
    filaments[shrinking_idx, 7] -= vs * dt


def find_close_fils(filaments, tip_pos, inhib_range, device="cpu"):
    """
    DEPRECATED: Extremely slow if used with GPUs. For CPU use only.
    Finds filaments that are within a specified inhibitory range of a given tip position.

    This function computes the minimum distance from the tip position to each filament,
    taking into account whether the closest point is behind the minus end, along the side,
    or beyond the plus end of the filament. It returns the subset of filaments that are
    closer than the specified inhibitory range.

    Args:
        filaments (torch.Tensor): Tensor of shape (N, 8), where each row represents a filament with:
            - columns 1:4: minus end position (x, y, z)
            - columns 4:7: unit direction vector
            - column 7: filament length
        tip_pos (torch.Tensor): Tensor of shape (3,) representing the (x, y, z) position of the tip.
        inhib_range (float): The inhibitory range threshold. Filaments closer than this distance are returned.
        device (str, optional): The device to perform computations on. Defaults to "cpu".

    Returns:
        torch.Tensor: Subset of `filaments` tensor containing only those filaments within `inhib_range` of `tip_pos`.
    """
    minus_end_to_tip_vec = tip_pos - filaments[:, 1:4]

    # Project tip position onto the filament direction vector
    fil_proj = torch.sum(minus_end_to_tip_vec * filaments[:, 4:7], dim=1)

    # Determine which filaments are behind, on the side, or in front of the tip
    behind_fil_idx = fil_proj < 0
    side_fil_idx = (0 < fil_proj) & (fil_proj < filaments[:, 7])
    infront_fil_idx = fil_proj > filaments[:, 7]

    dist_arr = torch.zeros(filaments.shape[0], device=device, dtype=torch.float64)

    dist_arr[behind_fil_idx] = torch.norm(
        minus_end_to_tip_vec[behind_fil_idx], dim=1
    ).to(torch.float64)

    dist_arr[side_fil_idx] = torch.norm(
        minus_end_to_tip_vec[side_fil_idx]
        - fil_proj[side_fil_idx, None] * filaments[side_fil_idx, 4:7],
        dim=1,
    ).to(torch.float64)

    # infront_fil_proj = torch.einsum("i,ij->ij", filaments[infront_fil_idx, 7], filaments[infront_fil_idx, 4:7])
    infront_fil_proj = (
        filaments[infront_fil_idx, 7].unsqueeze(1) * filaments[infront_fil_idx, 4:7]
    )
    dist_arr[infront_fil_idx] = torch.norm(
        minus_end_to_tip_vec[infront_fil_idx] - infront_fil_proj, dim=1
    ).to(torch.float64)

    close_fils = filaments[dist_arr < inhib_range]

    return close_fils


@cuda.jit(device=True)
def point_line_dist_sqr_device(point, line_seg):
    """point: 1D array of shape (3,)
    line: 1D array of shape (0,6) 0-2=minus end, 3-5=direction, 6=length
    return: distance float
    """
    minus_end = line_seg[:3]
    direction = line_seg[3:6]
    length = line_seg[6]

    minus_end_to_point = cuda.local.array(3, dtype=numba.float64)
    proj = 0.0
    for i in range(3):
        minus_end_to_point[i] = point[i] - minus_end[i]
        proj += minus_end_to_point[i] * direction[i]

    if proj < 0:
        dist_sqr = 0.0
        for i in range(3):
            dist_sqr += minus_end_to_point[i] * minus_end_to_point[i]
        return dist_sqr

    dist_sqr = 0.0
    if proj > length:
        for i in range(3):
            dist_dir = point[i] - (minus_end[i] + length * direction[i])
            dist_sqr += dist_dir * dist_dir
        return dist_sqr

    for i in range(3):
        dist_dir = minus_end_to_point[i] - proj * direction[i]
        dist_sqr += dist_dir * dist_dir

    return dist_sqr


@cuda.jit(device=True)
def safe_acos(value):
    return acos(max(-1.0, min(value, 1.0)))


@cuda.jit(device=True)
def calc_inhib_func(dir_tip, dir_fil, ang_senstv):
    dot_product = 0.0
    for i in range(3):
        dot_product += dir_tip[i] * dir_fil[i]
    return 1.0 / (1.0 + exp(-ang_senstv * (safe_acos(dot_product) - HALF_PI)))


@cuda.jit(device=True)
def get_catastrophe_rate_device(
    tip_idx, filaments, inhib_range, inhib_strength, ang_senstv
):
    k_cat = 0.0
    # Exit if filament was just nucleated, 2*inhib_range was arbitrarily chosen
    if filaments[tip_idx, 7] < 2.0 * inhib_range:
        return k_cat

    tip_pos = cuda.local.array(3, dtype=numba.float64)
    tip_fil = filaments[tip_idx]
    for i in range(3):
        tip_pos[i] = tip_fil[i + 1] + tip_fil[7] * tip_fil[i + 4]

    inhib_range_sqr = inhib_range**2
    for fil_idx in range(filaments.shape[0]):
        if fil_idx == tip_idx:  # Skip the current filament
            continue
        dr2 = point_line_dist_sqr_device(tip_pos, filaments[fil_idx][1:8])
        if dr2 < inhib_range_sqr:  # If the filament is within the inhibitory range
            # Calculate the contribution to the catastrophe rate
            # Note: tip_fil[4:7] is the direction vector of the growing filament
            k_cat += inhib_strength * calc_inhib_func(
                tip_fil[4:7], filaments[fil_idx][4:7], ang_senstv
            )
    assert k_cat >= 0
    return k_cat


@cuda.jit()
def catastrophize_kernel(
    filaments,
    not_shrinking_idx,
    kc,
    dt,
    inhib_range,
    inhib_strength,
    ang_senstv,
    rng_states,
):
    thread_id = cuda.grid(1)
    if thread_id < not_shrinking_idx.shape[0]:
        tip_idx = not_shrinking_idx[thread_id]
        k_inhib = get_catastrophe_rate_device(
            tip_idx, filaments, inhib_range, inhib_strength, ang_senstv
        )
        tot_cat_prob = min(1.0 - exp(-(k_inhib + kc) * dt), 1.0)
        unif = xoroshiro128p_uniform_float64(rng_states, thread_id)
        if unif < tot_cat_prob:
            filaments[tip_idx, 8] = -1.0


def check_gpu_limits(bpg, TPB):
    # Get device limits
    device = cuda.get_current_device()
    max_threads_per_block = device.MAX_THREADS_PER_BLOCK
    max_blocks_per_grid_x = device.MAX_GRID_DIM_X

    # Check if configuration is valid
    if TPB > max_threads_per_block:
        raise ValueError(
            f"Threads per block ({TPB}) exceeds the maximum allowed ({max_threads_per_block})."
        )

    if bpg > max_blocks_per_grid_x:
        raise ValueError(
            f"Blocks per grid ({bpg}) exceeds the maximum allowed ({max_blocks_per_grid_x})."
        )


def catastrophize_filaments(filaments, not_shrinking_idx, params, device="cpu"):
    # Set parameters
    ang_senstv = params["ang_senstv"] / torch.pi  # computational efficiency
    inhib_range = params["inhib_range"]
    inhib_strength = params["inhib_strength"]

    # Send arrays to device
    not_shrinking_idx = not_shrinking_idx.to(device)
    filaments = filaments.to(device)

    # If using the CUDA kernel
    if params.get("cuda", False):
        bpg = (not_shrinking_idx.shape[0] + TPB - 1) // TPB

        check_gpu_limits(bpg, TPB)

        rng_states = create_xoroshiro128p_states(
            TPB * bpg,
            seed=params["rng_seed"] + params["rng_iteration"],
            subsequence_start=params["rng_iteration"],
        )
        # Subsequence count is needed to prevent correlated rngs
        params["rng_iteration"] += 1

        # Note: Filament state is changed within the kernel
        catastrophize_kernel[bpg, TPB](
            filaments.numpy(),
            not_shrinking_idx.numpy(),
            params["kc"],
            params["dt"],
            inhib_range,
            inhib_strength,
            ang_senstv,
            rng_states,
        )
        cuda.synchronize()
        return

    ### If not using the CUDA kernel, use the CPU implementation ###
    k_arr = torch.ones(not_shrinking_idx.shape[0], device=device) * params["kc"]
    for i, g_fil in enumerate(filaments[not_shrinking_idx]):
        # Exit if filament just nucleated, 2*inhib_range arbitrarially chosen
        if g_fil[7] < 2 * inhib_range:
            continue
        fil_dir = g_fil[4:7]
        tip_pos = g_fil[1:4] + g_fil[7] * fil_dir
        # Find all filaments that are close to the tip
        close_fils = find_close_fils(filaments, tip_pos, inhib_range, device=device)
        if close_fils.shape[0] == 0:
            continue
        # Find the angle between the growing filament and the close filaments
        dot_product = torch.sum(fil_dir * close_fils[:, 4:7], dim=1)
        # TODO: Check if this is correct (probably will never use)
        ang_arr = torch.acos(torch.clamp(dot_product, -1, 1))
        k_arr[i] += (
            inhib_strength
            * (1.0 / (1.0 + torch.exp(-ang_senstv * (ang_arr - HALF_PI)))).sum()
        )

    prob_arr = 1.0 - torch.exp(-k_arr * params["dt"])
    cat_fil = prob_arr > torch.rand(prob_arr.shape[0])
    filaments[not_shrinking_idx[cat_fil], 8] = -1.0
    return


# Possibly unnecessary, should always be sorted
def search_for_filament_indices_by_id(filaments, bound_fil_ids):
    fil_ids = filaments[:, 0].contiguous()
    # Check to make sure ids are unique and sorted
    check_for_sorted_and_uniqueness(fil_ids)
    return torch.searchsorted(fil_ids, bound_fil_ids)


def find_filaments_out_of_bounds(growing_filaments, bounds):
    # Find position of growing filament tips
    tip_pos = (
        growing_filaments[:, 1:4]
        + growing_filaments[:, 7, None] * growing_filaments[:, 4:7]
    )

    out_of_bounds = torch.zeros(tip_pos.shape[0], dtype=bool)
    for dim, bound in enumerate(bounds):
        if bound > 0:
            out_of_bounds |= torch.abs(tip_pos[:, dim]) > bound

    return growing_filaments[out_of_bounds, 0]


def update_filament_states(filaments, growing_idx, paused_idx, params):
    # Sample poisson distribution for catastrophe
    not_shrinking_idx = torch.concatenate((growing_idx, paused_idx))

    # Pausing
    bounds = [
        params.get("Lx", -1) * 0.5,
        params.get("Ly", -1) * 0.5,
        params.get("Lz", -1) * 0.5,
    ]
    # If all bounds are non-positive, then there are no bounds and skip this step
    if not all(bound <= 0 for bound in bounds):
        # Find filaments that are outside the boundary
        pausing_fil_ids = find_filaments_out_of_bounds(filaments[growing_idx], bounds)
        # Get indices that match ids of filaments that are out of bounds
        pausing_fil_idx = search_for_filament_indices_by_id(filaments, pausing_fil_ids)
        filaments[pausing_fil_idx, 8] = 0

    # Get catastrophe rates of all growing filaments
    if params.get("inhib_range", 0) > 0 and params.get("inhib_strength", 0) > 0:
        catastrophize_filaments(filaments, not_shrinking_idx, params)

    else:  # Fast way if all catastrophe rates are equal
        # Get catastrophe probability assuming single Poisson distribution
        cat_prob = 1.0 - torch.exp(
            -torch.tensor(params["kc"], dtype=torch.float64) * params["dt"]
        )

        # Sample number of catastrophes from binomial distribution
        n_cat = torch.binomial(
            torch.tensor(not_shrinking_idx.shape[0], dtype=torch.float64),
            cat_prob,
        ).int()

        if n_cat > 0:
            cat_fil = torch.randperm(not_shrinking_idx.shape[0])[:n_cat]
            filaments[not_shrinking_idx[cat_fil], 8] = -1

    ########################

    # Removing filaments (length <= 0 and filament is depolymerizing)
    remove = filaments[:, 7] <= 0

    # Check for abberant filaments
    if torch.any(remove & (filaments[:, 8] >= 0)):
        print(
            "Warning: Filament with length <0 that is not depolymerizing at time step",
            file=sys.stderr,
        )
        print(filaments[remove & (filaments[:, 8] >= 0)], file=sys.stderr)
        # TODO: Save a file of the state when an error like this is thrown

    # TODO: This may be causing fragmentation of memory. Optimizing this could improve performance
    if torch.any(remove):
        unbind_nuc = filaments[remove & (filaments[:, 9] >= 0), 9].to(torch.int64)
        current_filaments = filaments[~remove]
        gc.collect()
        torch.cuda.empty_cache()
        return current_filaments, unbind_nuc

    unbind_nuc = torch.zeros(0, dtype=torch.int64)
    return filaments, unbind_nuc
