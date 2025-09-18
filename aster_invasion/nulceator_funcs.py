#!/usr/bin/env python

"""@package docstring
File: nucleator_funcs.py
Author: Adam Lamson
Email: alamson@flatironinstitute.org
Description:

Units: [F] = pN, [L] = um, [T] = s
"""

import torch
import numpy as np
from .filament_funcs import (
    search_for_filament_indices_by_id,
    check_for_sorted_and_uniqueness,
)


def update_nucleators(filaments, nucleators, unbind_nuc, params):
    tot_length = filaments[:, 7].sum()
    assert tot_length > 0, (
        f"Total filament length is less than or equal to zero. Total length = {tot_length}um "
    )

    ## Binding nucleators
    poss_binding_nuc = nucleators[nucleators[:, 1] < 0]
    poss_binding_nuc_idx = torch.where(nucleators[:, 1] < 0)[0]

    n_unbound_nuc = poss_binding_nuc.shape[0]

    prob_binding = 1.0 - torch.exp(-params["kb"] * params["dt"] * tot_length)
    # n_bind = np.random.binomial(n_unbound_nuc, prob_binding)
    n_bind = torch.binomial(
        torch.tensor(n_unbound_nuc, dtype=torch.float),
        prob_binding.float(),
    ).int()

    # Find filaments to bind to
    if n_bind > 0:
        bind_fil_idx = torch.multinomial(
            filaments[:, 7] / tot_length, n_bind, replacement=True
        )

        # Find locations for nucleators to bind
        nuc_bind_pos = torch.rand(n_bind) * filaments[bind_fil_idx, 7]

        # Set nucleator to bound
        nucleators[poss_binding_nuc_idx[:n_bind], 1:] = torch.stack(
            [filaments[bind_fil_idx, 0], torch.full((n_bind,), -1), nuc_bind_pos], dim=1
        ).type(torch.float64)

    ## Unbinding nucleators
    if unbind_nuc.shape[0] > 0:
        nucleators[unbind_nuc, 1:] = -1

    cur_bound_nuc = nucleators[nucleators[:, 1] >= 0]
    cur_bound_nuc_idx = torch.where(nucleators[:, 1] >= 0)[0]
    cur_bound_nuc_fil_idx = search_for_filament_indices_by_id(
        filaments, cur_bound_nuc[:, 1].contiguous()
    )
    cur_bound_nuc_fil_len = filaments[cur_bound_nuc_fil_idx, 7]
    nuc_detach_idx = cur_bound_nuc_idx[
        (cur_bound_nuc_fil_len - cur_bound_nuc[:, 3]) < 0
    ]

    if nuc_detach_idx.shape[0] > 0:
        nucleators[nuc_detach_idx, 1:] = -1


def nucleate_filaments_from_nucleators(filaments, nucleators, max_fil_id, params):
    # Find all nucleators that are bound to filaments but don't have a nucleated filament
    active_nuc = nucleators[(nucleators[:, 1] >= 0) & (nucleators[:, 2] < 0)]
    active_nuc_idx = torch.where((nucleators[:, 1] >= 0) & (nucleators[:, 2] < 0))[0]
    if active_nuc.shape[0] == 0:
        return filaments, max_fil_id
    # Find filaments that active nucleators are bound to
    bound_nuc_fil_idx = search_for_filament_indices_by_id(
        filaments, active_nuc[:, 1].contiguous()
    )

    # Start making new filaments
    new_fils = torch.zeros(
        (active_nuc.shape[0], filaments.shape[1]), device=filaments.device
    ).type(torch.float64)

    # Make new filament ids
    new_fil_ids = torch.arange(
        max_fil_id + 1,
        max_fil_id + np.float64(active_nuc.shape[0]) + 1,
        dtype=torch.float64,
    )

    check_for_sorted_and_uniqueness(new_fil_ids)
    # Get new filament minus ends
    new_fil_pos = filaments[bound_nuc_fil_idx, 1:4] + torch.einsum(
        "ij,i->ij", filaments[bound_nuc_fil_idx, 4:7], active_nuc[:, 3]
    )

    # Get filament directions dependent on dimensionality of the system
    new_fil_dir = (
        gen_rand_dir_vecs(
            filaments[bound_nuc_fil_idx, 4:7].to(torch.float64), params["std_ang"]
        )
        if params.get("dimension", 3) == 3
        else gen_rand_dir_vecs_2d(
            filaments[bound_nuc_fil_idx, 4:7].to(torch.float64), params["std_ang"]
        )
    )
    # Get centrosome id from MT that nucleated it
    cent_numbers = filaments[bound_nuc_fil_idx, 10]

    new_fil_len = torch.ones(active_nuc.shape[0]) * params["diam"]
    new_fil_state = torch.ones(active_nuc.shape[0])
    new_fils[:, 0] = new_fil_ids
    new_fils[:, 1:4] = new_fil_pos
    new_fils[:, 4:7] = new_fil_dir
    new_fils[:, 7] = new_fil_len
    new_fils[:, 8] = new_fil_state
    new_fils[:, 9] = active_nuc[:, 0]  # Nucleator id
    new_fils[:, 10] = cent_numbers
    nucleators[active_nuc_idx, 2] = new_fil_ids.type(torch.float64)

    # Cludge to work around torch single type arrays
    fils = torch.cat((filaments, new_fils)).type(torch.float64)
    max_fil_id += active_nuc.shape[0]

    return fils, max_fil_id


def rotation_matrix_from_vectors(
    vec1: torch.Tensor, vec2: torch.Tensor
) -> torch.Tensor:
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return rotation_matrix: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    vec1 = vec1.to(torch.float64)
    vec2 = vec2.to(torch.float64)

    a = vec1 / torch.norm(vec1)
    b = vec2 / torch.norm(vec2)
    c = torch.dot(a, b).type(torch.float64)
    if c > 0.999999:
        return torch.eye(3, dtype=torch.float64)
    if c < -0.999999:
        # Most orthogonal axis
        abs_a = torch.abs(a)
        min_idx = torch.argmin(abs_a)
        ortho = torch.zeros_like(a)
        ortho[min_idx] = 1.0
        axis = torch.linalg.cross(a, ortho)
        axis = axis / torch.norm(axis)
        return 2.0 * torch.outer(axis, axis) - torch.eye(3, dtype=torch.float64)

    v = torch.linalg.cross(a, b).type(torch.float64)
    s = torch.norm(v).type(torch.float64)
    kmat = torch.tensor([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    rotation_matrix = torch.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))
    return rotation_matrix


# @torch.jit.script
def gen_rand_dir_vecs(dir_vec: torch.Tensor, std_ang: float) -> torch.Tensor:
    theta_x = torch.normal(0, std_ang, size=(dir_vec.shape[0],))
    theta_y = torch.normal(0, std_ang, size=(dir_vec.shape[0],))
    x = torch.sin(theta_x)
    y = torch.sin(theta_y)
    z = torch.sqrt(1 - x**2 - y**2)
    random_vecs = torch.stack((x, y, z), dim=1).type(torch.float64)

    # Create an array of rotation matrices
    z_vec = torch.tensor([0.0, 0.0, 1.0])

    rot_matrices = torch.empty((dir_vec.shape[0], 3, 3)).type(torch.float64)
    for i in range(dir_vec.shape[0]):
        rot_matrices[i] = rotation_matrix_from_vectors(z_vec, dir_vec[i])

    # Apply the rotation to all vectors using explicit loops
    new_dir_vec = torch.empty_like(random_vecs).type(torch.float64)
    for i in range(dir_vec.shape[0]):
        new_dir_vec[i] = rot_matrices[i] @ random_vecs[i]

    new_dir_vec /= torch.norm(new_dir_vec, dim=1, keepdim=True)
    return new_dir_vec


def gen_rand_dir_vecs_2d(dir_vec: torch.Tensor, std_ang: float) -> torch.Tensor:
    # Get current angle of nucleator bounnd MTs
    cur_thetas = torch.arctan(dir_vec[:, 1] / dir_vec[:, 0])
    cur_thetas[dir_vec[:, 0] < 0] += torch.pi

    # Add angle perturbations for nucleated MTs
    new_thetas = cur_thetas + torch.normal(0, std_ang, size=(dir_vec.shape[0],))

    # Make new direction vectors
    x = torch.cos(new_thetas)
    y = torch.sin(new_thetas)
    z = torch.zeros(dir_vec.shape[0])
    new_dir_vec = torch.stack((x, y, z), dim=1)
    new_dir_vec /= torch.norm(new_dir_vec, dim=1, keepdim=True)
    return new_dir_vec
