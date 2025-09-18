#!/usr/bin/env python
"""@package docstring
File: aster_main.py
Author: Adam Lamson
Email: alamson@flatironinstitute.org
Description:

Units: [F] = pN, [L] = um, [T] = s
"""

import sys
from pathlib import Path
import torch
import numpy as np
import numba.cuda as cuda  # Needed to prevent segmentation fault
import h5py
import yaml
from time import time

from .arguments import parse_arguments
from .hdf5_to_vtk import convert_hdf5_to_vtk
from .filament_funcs import change_filament_lengths, update_filament_states
from .nulceator_funcs import update_nucleators, nucleate_filaments_from_nucleators
from .centrosome_funcs import nucleate_filaments_from_centrosomes


PARAMS = yaml.safe_load("""
t_total: 10 # Total time of simulation
t_snap: .1 # Time between snapshots
dt: .01 # Time step
rng_seed: 1234

# Box parameters
Lx: 10 # Box size in x [um]
Ly: 10 # Box size in y [um]
Lz: 10 # Box size in z [um]
dimension: 3

# Microtubule parameters
diam: .025 # Diameter of microtubules
vg: .35 # Growth velocity [um/s]
vs: .55 # Shrink velocity [um/s]
kc: 0.042 # Rate catastrophe [s^-1]
ang_senstv: 22. # Sensitivity to angle
## Turn on interactions
inhib_range: 2 # Range to calculate inhibition
inhib_strength: 10 # Strength of inhibition
## Turn off interactions
#inhib_range: .0 # Range to calculate inhibition
#inhib_strength: .0 # Strength of inhibition

# Nucleator parameters
n_nucs: 10000 # Number of nucleators
kb: .000047 # Nucleator binding rate [um^-1 s^-1]
std_ang: 0.15707963 # Standard deviation of nucleating angle [rad]

# Centrosome parameters
centrosome: True # Whether to include centrosomes
nuc_rate: .5 # Nucleation at centrosome rate [s^-1]

# Other parameters
cuda: False
file_name: "aster_invasion.h5" # File name to save data

""")


# MT data structure [
# 0id,
# 1minux x, 2minus y, 3minus z,
# 4direction x, 5direction y, 6direction z,
# 7length, 8state(-1=shrink, 0=paused, 1=grow), 9nucleator id(-1=none),
# 10centrosome id(-1=none)]
INIT_MTS = torch.tensor(
    [
        [0, 0.07185, 0.18665, 0.0, 0.35923, 0.93325, 0.0, 2.0, 1, -1, 0],
        [1, -0.14397, -0.13883, 0.0, -0.71983, -0.69415, 0.0, 2.0, 1, -1, 0],
        [2, -0.18489, 0.07627, 0.0, -0.92443, 0.38136, 0.0, 2.0, 1, -1, 0],
        [3, 0.04407, -0.19508, 0.0, 0.22034, -0.97542, 0.0, 2.0, 1, -1, 0],
        [4, 0.03745, -0.19646, 0.0, 0.18723, -0.98232, 0.0, 2.0, 1, -1, 0],
        [5, -0.02830, 0.19799, 0.0, -0.14148, 0.98994, 0.0, 2.0, 1, -1, 0],
        [6, -0.03310, 0.19724, 0.0, -0.16551, 0.98621, 0.0, 2.0, 1, -1, 0],
        [7, 0.06404, -0.18947, 0.0, 0.32018, -0.94736, 0.0, 2.0, 1, -1, 0],
        [8, 0.19312, -0.05200, 0.0, 0.96561, -0.26000, 0.0, 2.0, 1, -1, 0],
        [9, 0.14225, -0.14059, 0.0, 0.71124, -0.70295, 0.0, 2.0, 1, -1, 0],
        [10, -0.12536, 0.15584, 0.0, -0.62680, 0.77918, 0.0, 2.0, 1, -1, 0],
    ]
)

# Nucleator data structure [0id, 1MT bound, 2MT nucleated, 3MT bound arc
# position(<0 = unbound]
INIT_NUCLEATORS = torch.tensor(
    [
        [0, -1, -1, -1.0],
        [1, -1, -1, -1.0],
        [2, -1, -1, -1.0],
        [3, -1, -1, -1.0],
        [4, -1, -1, -1.0],
        [5, -1, -1, -1.0],
        [6, -1, -1, -1.0],
        [7, -1, -1, -1.0],
        [8, -1, -1, -1.0],
        [9, -1, -1, -1.0],
        [10, -1, -1, -1.0],
        [11, -1, -1, -1.0],
        [12, -1, -1, -1.0],
        [13, -1, -1, -1.0],
        [14, -1, -1, -1.0],
        [15, -1, -1, -1.0],
    ]
)

# Centrosome data structure [0id, 1x, 2y, 3z, 4radius]
INIT_CENTROSOMES = torch.tensor(
    [
        [0, 0, 0, 0, 0.2],
    ]
)


def init_system(init_mts, init_nucleators):
    """!Initialize the system
    @return: void
    """
    filaments = init_mts.clone()
    if filaments.shape[0] == 0:
        filaments = torch.zeros((0, 11))
    # Make sure filament lengths are correct
    nucleators = init_nucleators.clone()
    return (
        filaments,
        nucleators,
    )


def create_nucleators(n_nucs):
    """!Create nucleators
    @return: void
    """
    nucs = torch.zeros((n_nucs, 4), dtype=torch.float64)
    nucs[:, 0] = torch.arange(n_nucs)
    nucs[:, 1:3] = -1
    nucs[:, 3] = -1.0
    return nucs


def write_out_system(t, h5_file, filaments, nucleators, centrosomes):
    """!Write out the system
    @return: void
    """
    # TODO: Make this easy to restart, open up file every time, then close
    grp = h5_file.create_group(f"t_{t}")

    fil_dset = grp.create_dataset("filaments", data=filaments.numpy())
    fil_dset.attrs["total_length"] = filaments[:, 7].sum().item()

    nuc_dset = grp.create_dataset("nucleators", data=nucleators.numpy())
    cent_dset = grp.create_dataset("centrosomes", data=centrosomes.numpy())


# Cludgy way to update parameters, make this better if other people are
# going to use it
def update_params(args):
    print(Path.cwd())
    with args.filepath.open("r") as f:
        print(f"Loading parameters from {sys.argv[1]}")
        PARAMS = yaml.safe_load(f)
    if PARAMS.get("mt_file", 0):
        INIT_MTS = torch.tensor(np.loadtxt(PARAMS["mt_file"], delimiter=",", ndmin=2))
    if PARAMS.get("dimension", 3) == 2:
        # If two dimensions, get rid of z-direction and renormalize directions
        INIT_MTS[:, 3] = 0
        INIT_MTS[:, 6] = 0
        INIT_MTS[:, 4:7] /= torch.norm(INIT_MTS[:, 4:7], dim=1).view(-1, 1)

    INIT_MTS = INIT_MTS.to(torch.float64)

    if PARAMS.get("nuc_file", 0):
        INIT_NUCLEATORS = torch.tensor(
            np.loadtxt(PARAMS["nuc_file"], delimiter=",", ndmin=2), dtype=torch.float64
        )
    elif PARAMS.get("n_nucs", 0):
        INIT_NUCLEATORS = create_nucleators(PARAMS["n_nucs"]).type(torch.float64)
    else:
        INIT_NUCLEATORS = torch.zeros((0, 4)).type(torch.float64)

    if not PARAMS.get("centrosome", True):
        # If no centrosomes, set to empty array
        INIT_CENTROSOMES = torch.zeros((0, 5))
    elif PARAMS.get("centrosome_file", 0):
        # Load this as a multi-dimensional array even if there is only one
        # centrosome
        INIT_CENTROSOMES = torch.tensor(
            np.loadtxt(PARAMS["centrosome_file"], delimiter=",", ndmin=2)
        )

    PARAMS["rng_seed"] = PARAMS.get("rng_seed", 1234)
    # Set random seed
    torch.manual_seed(PARAMS["rng_seed"])
    torch.cuda.manual_seed(PARAMS["rng_seed"])
    PARAMS["rng_iteration"] = PARAMS.get("rng_iteration", 0)
    PARAMS["n_steps_snap"] = int(PARAMS["t_snap"] / PARAMS["dt"])

    return PARAMS, INIT_MTS, INIT_NUCLEATORS, INIT_CENTROSOMES


def main():
    """!Main function for the program
    @return: void
    """
    args = parse_arguments()
    if args.hdf5_to_vtk:
        print("Converting hdf5 to vtk")
        convert_hdf5_to_vtk(args.filepath)
        sys.exit()

    PARAMS, INIT_MTS, INIT_NUCLEATORS, INIT_CENTROSOMES = update_params(args)
    time_start = time()

    filaments, nucleators = init_system(INIT_MTS, INIT_NUCLEATORS)
    max_fil_id = np.float64(
        -1 if filaments.shape[0] == 0 else filaments[:, 0].max().item()
    )

    # Open file to write data
    h5_file = h5py.File(PARAMS["file_name"], "w")

    # Save parameters to attrs for easier analysis
    h5_file.attrs["params"] = yaml.dump(PARAMS)
    write_out_time_start = time()
    loop_times = []

    # Main sim loop
    for i, t in enumerate(np.arange(0, PARAMS["t_total"], PARAMS["dt"])):
        loop_start_time = time()

        # If no filaments exist, try to nucleate from centrosomes until there are
        if max_fil_id == -1 and INIT_CENTROSOMES.shape[0] > 0:
            filaments, max_fil_id = nucleate_filaments_from_centrosomes(
                filaments, INIT_CENTROSOMES, max_fil_id, PARAMS
            )
            continue

        growing = filaments[:, 8] == 1
        shrinking = filaments[:, 8] == -1
        paused = filaments[:, 8] == 0

        growing_idx = torch.where(growing)[0]
        shrinking_idx = torch.where(shrinking)[0]
        paused_idx = torch.where(paused)[0]

        change_filament_lengths(filaments, growing_idx, shrinking_idx, PARAMS)

        if filaments.shape[0] <= 0 and INIT_CENTROSOMES.shape[0] == 0:
            print("No filaments left in simulation")
            break

        filaments, unbind_nuc = update_filament_states(
            filaments, growing_idx, paused_idx, PARAMS
        )

        update_nucleators(filaments, nucleators, unbind_nuc, PARAMS)
        # Init filaments from centrosomes
        if INIT_CENTROSOMES.shape[0] > 0:
            filaments, max_fil_id = nucleate_filaments_from_centrosomes(
                filaments, INIT_CENTROSOMES, max_fil_id, PARAMS
            )

        filaments, max_fil_id = nucleate_filaments_from_nucleators(
            filaments, nucleators, max_fil_id, PARAMS
        )

        # Write out system
        if (i % PARAMS["n_steps_snap"]) == 0:
            loop_times.append(time() - loop_start_time)
            iter_time = time() - write_out_time_start
            print(
                f"sim time: {t} sec, walltime {time() - time_start:.4g} sec (writeout {iter_time:.4g} sec)"
            )
            write_out_system(t, h5_file, filaments, nucleators, INIT_CENTROSOMES)
            write_out_time_start = time()
            sys.stdout.flush()

    time_end = time()
    print("Time to run simulation: ", time_end - time_start)
    h5_file.attrs["wall_time"] = time_end - time_start
    h5_file.create_dataset("loop_times", data=loop_times)
    h5_file.close()


# ##########################################
if __name__ == "__main__":
    print("Running main")
    main()

# ##########################################
