#!/usr/bin/env python

"""@package docstring
File: centrosome_funcs.py
Author: Adam Lamson
Email: alamson@flatironinstitute.org
Description:

Units: [F] = pN, [L] = um, [T] = s
"""

import torch
from .filament_funcs import check_for_sorted_and_uniqueness


def nucleate_filaments_from_centrosomes(filaments, centrosomes, max_fil_id, params):
    # Sample how many filaments to nucleate from each centrosome
    surface_area_arr = 4.0 * torch.pi * centrosomes[:, 4] ** 2
    avg_nuc = (
        params["nuc_rate"]
        * params["dt"]
        * centrosomes.shape[0]
        * surface_area_arr.sum()
    )
    n_nuc_fils = int(torch.poisson(avg_nuc).item())
    if n_nuc_fils == 0:
        return filaments, max_fil_id

    cent_ind_to_nuc = torch.multinomial(
        surface_area_arr / surface_area_arr.sum(),
        n_nuc_fils,
        replacement=True,
    )
    new_fils = torch.zeros(
        (n_nuc_fils, filaments.shape[1]), device=filaments.device
    ).type(torch.float64)

    new_fil_ids = torch.arange(
        max_fil_id + 1, max_fil_id + n_nuc_fils + 1, dtype=torch.float64
    )

    check_for_sorted_and_uniqueness(new_fil_ids)

    # Get random directions to nucleate filaments
    thetas = torch.rand(n_nuc_fils) * 2 * torch.pi
    if params.get("dimension", 3) == 3:
        phis = torch.acos(2 * torch.rand(n_nuc_fils) - 1)
        new_fil_dir = torch.stack(
            (
                torch.sin(phis) * torch.cos(thetas),
                torch.sin(phis) * torch.sin(thetas),
                torch.cos(phis),
            )
        ).T
    else:
        new_fil_dir = torch.stack(
            (torch.sin(thetas), torch.cos(thetas), torch.zeros(n_nuc_fils))
        ).T

    # Get position from random directions
    cent_pos = centrosomes[cent_ind_to_nuc, 1:4]
    cent_rads = centrosomes[cent_ind_to_nuc, 4]
    new_fil_pos = cent_rads[:, None] * new_fil_dir + cent_pos
    new_fil_state = torch.ones(n_nuc_fils)
    new_fil_len = torch.ones(n_nuc_fils) * params["diam"]
    new_fil_nucs = -1 * torch.ones(n_nuc_fils)  # Not created by nucleators
    new_fil_cent_id = centrosomes[cent_ind_to_nuc, 0]

    new_fils[:, 0] = new_fil_ids
    new_fils[:, 1:4] = new_fil_pos
    new_fils[:, 4:7] = new_fil_dir
    new_fils[:, 7] = new_fil_len
    new_fils[:, 8] = new_fil_state
    new_fils[:, 9] = new_fil_nucs  # Nucleator id
    new_fils[:, 10] = new_fil_cent_id  # Centrosome id

    fils = torch.cat((filaments, new_fils)).type(torch.float64)
    return fils, max_fil_id + n_nuc_fils
